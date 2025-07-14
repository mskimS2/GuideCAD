import random
import os
import numpy as np
from transformers import GPT2Tokenizer


def vector_to_text(commands):
    result = []
    for command in commands:
        cmd_type = command[0]
        values = command[1:]

        ALL_COMMANDS = ["Line", "Arc", "Circle", "EOS", "SOL", "Ext"]
        LINE_IDX = ALL_COMMANDS.index("Line")
        ARC_IDX = ALL_COMMANDS.index("Arc")
        CIRCLE_IDX = ALL_COMMANDS.index("Circle")
        EOS_IDX = ALL_COMMANDS.index("EOS")
        SOL_IDX = ALL_COMMANDS.index("SOL")
        EXT_IDX = ALL_COMMANDS.index("Ext")

        if cmd_type == LINE_IDX:  # Line
            x, y = values[:2]
            result.append(f"L {x} {y}")

        elif cmd_type == ARC_IDX:  # Arc
            x, y, α, f = values[:4]
            result.append(f"A {x} {y} {α} {f}")

        elif cmd_type == CIRCLE_IDX:  # Circle
            x, y, _, _, r = values[:5]
            result.append(f"R {x} {y} {r}")

        elif cmd_type == EXT_IDX:  # Extrude
            θ, φ, γ, px, py, pz, s, e1, e2, b, u = values[5:]
            result.append(f"E {θ} {φ} {γ} {px} {py} {pz} {s} {e1} {e2} {b} {u}")

        elif cmd_type == SOL_IDX:  # Start of Sequence
            result.append("SOL")

        elif cmd_type == EOS_IDX:  # End of Sequence
            result.append("EOS")

        else:
            raise ValueError(f"Unknown command type: {cmd_type}")

    return "\n".join(result)


def text_to_padded_vector(text, padding_value=-1, vector_length=17):
    # Map command types to numeric codes
    command_mapping = {
        "L": 0,  # Line
        "A": 1,  # Arc
        "R": 2,  # Circle
        "EOS": 3,
        "SOL": 4,
        "E": 5,  # Extrude
    }

    commands = text.split("\n")
    result = []

    def set_params(cmd_type, params):
        if cmd_type == "L":  # Line
            x, y = params
            return [x, y] + [padding_value] * (vector_length - 3)
        elif cmd_type == "A":  # Arc
            x, y, α, f = params
            return [x, y, α, f] + [padding_value] * (vector_length - 5)
        elif cmd_type == "R":  # Circle
            x, y, r = params
            return [x, y, -1, -1, r] + [padding_value] * (vector_length - 6)
        elif cmd_type == "E":  # Extrude
            θ, φ, γ, px, py, pz, s, e1, e2, b, u = params
            return [padding_value] * (vector_length - 12) + [θ, φ, γ, px, py, pz, s, e1, e2, b, u]
        else:
            raise ValueError(f"Unsupported command type: {cmd_type}")

    for cmd in commands:

        if cmd in command_mapping:
            vector = [command_mapping[cmd]] + [padding_value] * (vector_length - 1)
        elif cmd.startswith(("L ", "A ", "R ", "E ")):
            cmd_type = cmd[0]
            params = list(map(int, cmd[2:].split(" ")))
            vector = [command_mapping[cmd_type]] + set_params(cmd_type, params)
        else:
            raise ValueError(f"Unknown command format: {cmd}")

        result.append(vector)

    return np.array(result, dtype=int)


def extract_folder_and_files(directory):
    folder_list = []
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            file_list = os.listdir(folder_path)
            folder_list.append({"folder": folder, "files": file_list})

    return folder_list


def print_folder_data(folder_data):
    for folder_data in folder_data:
        print(f"폴더명: {folder_data['folder']}")
        for file in folder_data["files"]:
            print(f"  - 파일명: {file}")


def create_directory_if_not_exists(image_root):
    if not os.path.exists(image_root):
        os.makedirs(image_root)


def save_as_text(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def generate_cad_description(commands, include_loop_intro=True, mask_mode=False):
    grouped_descriptions = []
    loop_count = 0  # Counter for "Loop1", "Loop2", etc.

    introduces = [
        "Let's start by building the foundation of the design.",
        "We begin by sketching the basic structure.",
        "Starting with key shapes helps define the layout.",
        "The design starts to take form with simple elements.",
        "We lay out the initial framework to guide the model.",
        "Sketching lines and curves gives the model its foundation.",
        "This step establishes the symmetry and balance of the design.",
        "We start shaping the model with basic geometric features.",
        "Building a clear framework sets the tone for the design.",
        "The design begins to take shape with well-placed elements.",
    ]

    # Word-level variations for diverse expression
    word_variations = {
        "arc": ["graceful arc", "elegant curve", "sweeping path", "gentle bend"],
        "line": ["straight connection", "linear segment", "precise path", "defined edge"],
        "circle": ["circular feature", "rounded element", "symmetrical curve", "curved form"],
        "extrusion": ["structured extension", "volumetric projection", "3D extension", "solid form"],
        "direction": ["clockwise", "counter-clockwise"],
        "geometry": ["length", "area", "volume", "dimensions", "proportions"],
        "scale": ["scaled by", "adjusted to", "set at"],
        "projection": ["extending outward", "projecting symmetrically", "stretching"],
    }

    # Grouped templates with level-specific detail
    grouped_templates = {
        "L": [
            "Let's connect {count} lines to form a framework at {details}.",
            "Using {count} lines, we define the framework at {details}.",
            "We use {count} lines to connect key points like {details}.",
            "With {count} lines, we form the edges at {details}.",
            "We use {count} lines to make smooth transitions in this stage.",
            "The structure comes together with {count} lines at {details}.",
            "{count} lines complete the structure at {details}.",
        ],
        "A": [
            "We connect key points at {details} using {count} smooth arcs.",
            "Let's draw {count} arcs to create fluid curves around {details}.",
            "Using {count} arcs, we add elegance to the layout at {details}.",
            "We shape the structure with {count} gentle arcs near {details}.",
            "Drawing {count} arcs at {details} helps refine the design's flow.",
            "With {count} arcs, we ensure a smooth transition at {details}.",
            "We add {count} graceful arcs to guide the design near {details}.",
            "{count} arcs flow naturally around {details}, enhancing the layout.",
            "Curves made with {count} arcs bring balance to {details}.",
            "Let's enhance the geometry by sketching {count} arcs at {details}.",
        ],
        "R": [
            "We add {count} circular shapes near {details} to enhance symmetry.",
            "Drawing {count} circles around {details} helps balance the layout.",
            "Let's position {count} circles at {details} to refine the proportions.",
            "{count} circular features near {details} bring structure and clarity.",
            "To improve harmony, we place {count} circles at {details}.",
            "Adding {count} precise circles at {details} anchors the design.",
            "We create smooth transitions by adding {count} circles around {details}.",
            "The layout is enhanced with {count} circular elements near {details}.",
            "Positioning {count} circles at {details} ensures a balanced structure.",
            "The geometry gains clarity with {count} circles placed near {details}.",
        ],
        "E": [
            "Start at {start} on a sketch plane oriented ({theta}, {phi}, {gamma}) with origin at ({px}, {py}, {pz}). Create {count} extrusions scaled by {scale_value}, extending to {e1} and {e2} with {extrude_type} extrusion and {boolean} operation.",
            "From {start} on a sketch plane ({theta}, {phi}, {gamma}) at origin ({px}, {py}, {pz}), draw {count} extrusions scaling to {scale_value} and extending to {e1} and {e2} as {extrude_type} with {boolean} operation.",
            "Create {count} extrusions at {start} on a sketch plane ({theta}, {phi}, {gamma}) with origin ({px}, {py}, {pz}), scaling by {scale_value} to reach {e1} and {e2}. Extrusion: {extrude_type}, Operation: {boolean}.",
            "Draw {count} extrusions starting at {start} on a sketch plane ({theta}, {phi}, {gamma}) with origin at ({px}, {py}, {pz}). Scale {scale_value}, extend to {e1} and {e2}, and use {extrude_type} extrusion with {boolean} operation.",
            "Begin {count} extrusions from {start} on a sketch plane ({theta}, {phi}, {gamma}) at origin ({px}, {py}, {pz}). Scale to {scale_value}, project to {e1} and {e2}, using {extrude_type} extrusion and {boolean} operation.",
            "{count} extrusions start at {start}, on a sketch plane ({theta}, {phi}, {gamma}), with origin ({px}, {py}, {pz}). Scale: {scale_value}, Extend: {e1}, {e2}, Type: {extrude_type}, Operation: {boolean}.",
            "Add {count} extrusions at {start} on a sketch plane ({theta}, {phi}, {gamma}) with origin ({px}, {py}, {pz}). Scale by {scale_value}, extend to {e1} and {e2}, using {extrude_type} type and {boolean} operation.",
            "Create {count} extrusions starting from {start}, on a sketch plane ({theta}, {phi}, {gamma}), with origin ({px}, {py}, {pz}). Scale {scale_value}, extend to {e1} and {e2}, and apply {extrude_type} extrusion with {boolean} operation.",
            "At {start} on a sketch plane ({theta}, {phi}, {gamma}) with origin ({px}, {py}, {pz}), draw {count} extrusions scaled by {scale_value} to reach {e1} and {e2}. Extrusion type: {extrude_type}, Boolean operation: {boolean}.",
            "Draw {count} extrusions starting from {start} on a sketch plane ({theta}, {phi}, {gamma}) at origin ({px}, {py}, {pz}), scaled to {scale_value}, extending to {e1} and {e2}, with {extrude_type} type and {boolean} operation.",
        ],
    }

    # Loop description templates
    loop_templates = [
        "Shape {count} brings together curves and edges to balance the design.",
        "In Shape {count}, we refine the transitions to make the structure smooth.",
        "Shape {count} improves the design by combining arcs and lines.",
        "This stage in Shape {count} focuses on symmetry and alignment for a clean look.",
        "In Shape {count}, soft curves and sharp lines come together seamlessly.",
        "Paths in Shape {count} are adjusted to create better flow and balance.",
        "Shape {count} enhances the structure by blending arcs and straight lines.",
        "This phase in Shape {count} adds clarity with carefully defined paths.",
        "Curves in Shape {count} give the design a smooth and fluid motion.",
        "Shape {count} aligns key elements for a polished and cohesive layout.",
    ]

    def describe_grouped_command(command, params_list):
        if not params_list:  # Handle empty group
            return ""

        count = len(params_list)  # Define 'count' here

        if command == "L":  # Lines
            # Add natural flow to point descriptions
            points = []
            for idx, p in enumerate(params_list):
                point = f"({p[0]}, {p[1]})"
                if idx == len(params_list) - 1:
                    points.append(f"and {point}")
                else:
                    points.append(point)
            details = ", ".join(points)
            template = random.choice(grouped_templates["L"])
            return template.format(count=count, details=details)

        elif command == "A":  # Arcs
            details = []
            for p in params_list:
                point = f"({p[0]}, {p[1]})"
                angle = p[2]
                direction = "clockwise" if p[3] == 1 else "counter-clockwise"
                details.append(f"an arc ending at {point}, sweeping {angle}° in a {direction} direction")
            template = random.choice(grouped_templates["A"])
            return template.format(count=count, details=", ".join(details))

        elif command == "R":  # Circles
            details = []
            for p in params_list:
                center = f"({p[0]}, {p[1]})"
                radius = p[2]
                details.append(f"a circle centered at {center} with a radius of {radius}")
            template = random.choice(grouped_templates["R"])
            return template.format(count=count, details=", ".join(details))

        elif command == "E":  # Extrusions
            details = []
            extrude_types = {0: "one-sided", 1: "symmetric", 2: "two-sided"}
            boolean_operations = {0: "creating a new body", 1: "joining", 2: "cutting", 3: "intersecting"}

            for p in params_list:
                # Parse parameters
                theta, phi, gamma = p[0], p[1], p[2]  # Sketch plane orientation
                px, py, pz = p[3], p[4], p[5]  # Sketch plane origin
                scale_value = p[6]  # Scale
                e1, e2 = p[7], p[8]  # Extrude distances
                boolean = boolean_operations.get(p[9], "unknown operation")  # Boolean type
                extrude_type = extrude_types.get(p[10], "unknown type")  # Extrude type

                start = f"({px}, {py}, {pz})"  # Using origin as starting point for description

            # Choose a template
            template = random.choice(grouped_templates["E"])

            # Format the template
            return template.format(
                count=len(params_list),  # Number of extrusions
                start=start,
                theta=theta,
                phi=phi,
                gamma=gamma,
                px=px,
                py=py,
                pz=pz,
                scale_value=scale_value,
                e1=e1,
                e2=e2,
                extrude_type=extrude_type,
                boolean=boolean,
            )
        else:
            return f"Command {command} with {count} elements is not supported."

            return f"Command {command} with {len(params_list)} elements is not supported."

    # Process commands
    last_command = None
    grouped_commands = []
    for line in commands:
        line = line.strip()
        if line == "SOL":
            if grouped_commands:
                description = describe_grouped_command(last_command, grouped_commands)
                if include_loop_intro:
                    loop_intro = random.choice(loop_templates).format(count=loop_count, description=description)
                    if mask_mode:
                        loop_intro = random.choice(loop_templates).format(count="[MASK]", description="[MASK]")
                    grouped_descriptions.append(loop_intro)
                grouped_descriptions.append(description)
                grouped_commands = []
            loop_count += 1
            continue
        elif line.startswith("EOS"):
            if grouped_commands:
                description = describe_grouped_command(last_command, grouped_commands)
                grouped_descriptions.append(description)
            break
        else:
            parts = line.split()
            command = parts[0]
            params = list(map(int, parts[1:]))

            # If the command matches the last one, group it
            if command == last_command:
                grouped_commands.append(params)
            else:
                if grouped_commands:
                    description = describe_grouped_command(last_command, grouped_commands)
                    grouped_descriptions.append(description)
                last_command = command
                grouped_commands = [params]

    return random.choice(introduces) + " " + " ".join(grouped_descriptions)


if __name__ == "__main__":
    cad_commands = """
    SOL
    R 176 128 48
    E 192 128 192 128 121 128 14 209 128 0 0
    SOL
    R 176 128 48
    SOL
    R 176 128 32
    E 192 128 192 128 117 128 21 130 128 1 0
    SOL
    L 176 101
    L 223 128
    L 223 183
    L 176 210
    L 128 183
    L 128 128
    SOL
    R 176 155 25
    E 192 128 192 128 155 201 27 139 128 0 0
    EOS
    """.strip().split(
        "\n"
    )

    # Generate descriptions
    text = generate_cad_description(cad_commands, include_loop_intro=True)
    print(text)

    from transformers import GPT2Tokenizer

    # Initialize GPT2 tokenizer
    max_length = 512
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    print(tokenized["input_ids"].shape)
