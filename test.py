import time
import torch
import numpy as np
import h5py
from tqdm import tqdm
from cadlib.macro import *
from dataset.cad_dataset import *
from models.guidecad import GuideCAD
from cadlib.visualize import *
from loss import CADLoss


def logits2vec(outputs, refill_pad=True, to_numpy=True, device="cuda:0"):
    out_command = torch.argmax(torch.softmax(outputs["command_logits"], dim=-1), dim=-1)  # (N, S)
    out_args = torch.argmax(torch.softmax(outputs["args_logits"], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
    if refill_pad:  # fill all unused element to -1
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().to(device)[out_command.long()]
        out_args[mask] = -1

    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec


@torch.inference_mode()
def inference(model, test_loader, loss_fn, device, output_path: str):
    model.eval()
    total_loss = 0.0

    for data in tqdm(test_loader, ncols=120):
        gt_command = data["command"].to(device)
        gt_param = data["args"].to(device)
        prefix = data["prefix"].to(device)
        input_ids = data["input_ids"].to(device)
        mask = data["mask"].to(device)

        command_logits, args_logits, text_logits = model(input_ids=input_ids, attention_mask=mask, img_embed=prefix)

        loss_dict = loss_fn(
            command_logits, args_logits, gt_command, gt_param, text_logits[:, model.prefix_length :, :], input_ids
        )

        batch_out_vec = logits2vec({"command_logits": command_logits, "args_logits": args_logits}, device=device)
        gt_vec = torch.cat([gt_command.unsqueeze(-1), gt_param], dim=-1).detach().cpu().numpy()

        batch_size = gt_command.shape[0]
        for j in range(batch_size):
            out_vec = batch_out_vec[j]
            seq_len = gt_command[j].tolist().index(EOS_IDX)

            data_id = data["id"][j].split("/")[-1]

            with h5py.File(output_path + f"/{data_id}.h5", "w") as fp:
                fp.create_dataset("pred_vec", data=out_vec[:seq_len], dtype=np.int32)
                fp.create_dataset("gt_vec", data=gt_vec[j][:seq_len], dtype=np.int32)

        loss = loss_dict["loss_cmd"] + loss_dict["loss_args"]
        total_loss += loss.item()

    return total_loss


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ckpt", required=True, help="Path to the trained model checkpoint.")
    args.add_argument(
        "--root_path", required=True, help="Path where the predicted CAD sequences will be saved as .h5 files."
    )
    args.add_argument("--device", default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    p = args.parse_args()
    print("p", p.config)

    device = p.device
    root_path = p.root_path
    ckpt = torch.load(p.ckpt)
    print("ckpt.keys()", ckpt.keys())

    config = ckpt["config"]
    print("config", config)

    tokenizer = setup_tokenizer()

    # create dataloader
    dataset = CADDataset("test", config, tokenizer, config.model.prefix_mode)
    test_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=config.model.num_workers,
        worker_init_fn=np.random.seed(),
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2,
        persistent_workers=(True if config.model.num_workers > 0 else False),
    )

    # device setup

    loss_fn = CADLoss(config, device, tokenizer).to(device)

    model = GuideCAD(
        prefix_embed_dim=config.model.img_embed_dim,
        num_commands=config.cad_params.num_commands,
        num_params=config.cad_params.num_params,
        param_range=config.cad_params.param_range,
        prefix_length=config.model.prefix_length,
        d_model=config.model.d_model,
        dim_z=config.model.dim_z,
        adapt_embed=config.model.adapt_embed,
        finetune=config.model.finetune,
    ).to(device)
    model.eval()
    model.gpt.resize_token_embeddings(len(tokenizer))

    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()})

    start = time.perf_counter()
    total_loss = inference(model, test_loader, loss_fn, device, root_path + "/h5")
    end = time.perf_counter()
    print(f"execution time: {end - start:.2f} sec")

    print("total_loss", total_loss)
