import torch
import torch.nn.functional as F
from torch import nn
from cadlib.macro import CMD_ARGS_MASK, EOS_IDX


class CADLoss(nn.Module):
    def __init__(self, config, device, tokenizer, use_text: bool = False):
        super().__init__()
        self.device = device
        self.n_commands = config.cad_params.num_commands
        self.args_dim = config.cad_params.args_dim + 1
        self.cmd_weight = config.loss_weights.cmd_weight
        self.args_weight = config.loss_weights.args_weight
        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK, device=device))

        self.use_text = use_text
        self.pad_token_id = tokenizer.pad_token_id
        self.text_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    def forward(self, command_logits, args_logits, gt_command, gt_param, text_logits=None, gt_text=None):
        visibility_mask = self._get_visibility_mask(gt_command, seq_dim=-1)
        padding_mask = self._get_padding_mask(gt_command, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        mask = self.cmd_args_mask[gt_command.long()].to(self.device)

        loss_cmd = (
            F.cross_entropy(
                command_logits[padding_mask.bool()].reshape(-1, self.n_commands),
                gt_command[padding_mask.bool()].reshape(-1).long(),
            )
            * self.cmd_weight
        )

        loss_args = (
            F.cross_entropy(
                args_logits[mask.bool()].reshape(-1, self.args_dim), gt_param[mask.bool()].reshape(-1).long() + 1
            )
            * self.args_weight
        )  # shift due to -1 PAD_VAL

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}

        if self.use_text and not text_logits and not gt_text:
            loss_text = self.text_loss(
                text_logits[:, :-1, :].contiguous().view(-1, text_logits.size(-1)), gt_text[:, 1:].contiguous().view(-1)
            )

            res["loss_text"] = loss_text

        return res

    @staticmethod
    def _compute_cross_entropy(logits, targets, mask, num_classes, weight):
        logits_masked = logits[mask.bool()].reshape(-1, num_classes)
        targets_masked = targets[mask.bool()].reshape(-1).long()
        return weight * F.cross_entropy(logits_masked, targets_masked)

    @torch.no_grad()
    def _get_visibility_mask(self, commands, seq_dim=0):
        """
        Args:
            commands: Shape [S, ...]
        """
        S = commands.size(seq_dim)
        visibility_mask = (commands == EOS_IDX).sum(dim=seq_dim) < S - 1

        if seq_dim == 0:
            return visibility_mask.unsqueeze(-1)
        return visibility_mask

    @torch.no_grad()
    def _get_padding_mask(self, commands, seq_dim=0, extended=False):
        padding_mask = (commands == EOS_IDX).cumsum(dim=seq_dim) == 0
        padding_mask = padding_mask.float()

        if extended:
            # padding_mask doesn't include the final EOS, extend by 1 position to include it in the loss
            S = commands.size(seq_dim)
            torch.narrow(padding_mask, seq_dim, 3, S - 3).clone().add_(
                torch.narrow(padding_mask, seq_dim, 0, S - 3)
            ).clamp_(max=1)

        if seq_dim == 0:
            return padding_mask.unsqueeze(-1)
        return padding_mask
