from __future__ import annotations

from PIL import Image


PROMPTS = {
    "guided": {
        "system": (
            "You are a language translation assistant for robot manipulation tasks. "
            "You translate human instructions into commands a Vision-Language-Action (VLA) robot can act on.\n"
            "You will be given:\n"
            "- A user-provided instruction describing a manipulation goal and an image of the scene.\n"
            "Your task is to:\n"
            "1. Understand the meaning of the original instruction.\n"
            "2. Correctly identify the object to pick and the target location to place it, "
            "then reason about how the robot visually perceives each — "
            "these may not be the same as your own understanding.\n"
            "3. Generate a rephrased command that helps the robot locate and act on the target.\n"
            "Keep in mind: the VLA perceives the scene visually and may not recognize objects "
            "the same way you do. Your translation must bridge that perceptual gap.\n"
            "Feel free to explore.\n"
        ),
        "user": (
            "Task: {instruction}.\n"
            "Generate a command for the robot to successfully complete the task.\n"
            "When describing the object or the target, always reason from what the robot likely "
            "sees in the image, not from your own understanding of the object. "
            "The rules below are all applications of this.\n"
            "Rules for the command:\n"
            "1. Always keep the object name (e.g. 'carrot'), "
            "unless the object is unrecognizable to the robot — in that case replace the name "
            "with a visual description per Rule 7.\n"
            "2. One short sentence. No line breaks, no numbering.\n"
            "3. Try different grounding cues to help locate the object or the target — for example: "
            "spatial words (left, right, center, far, near), "
            "visual properties (color, size, shape), "
            "or relative position (closest, next to the cup). "
            "Try combining cues (e.g. 'far right', 'small red one on the left')."
            "Feel free to totaly change the name of the target if it may resemble more simple, real thing."
            "4. Use only simple, everyday words.\n"
            "5. If the object to pick or the placement target is something the robot is unlikely "
            "to recognize by name (abstract concept, brand, celebrity, symbol), describe it purely "
            "by its visual appearance in the scene — what the robot would actually see it as.\n"
        ),
    },
    "simple": {
        "system": (
            "You are a language translation assistant for robot manipulation tasks. You translate human instructions into commands a Vision-Language-Action (VLA) robot can act on.\n"
            "You will be given:\n"
            "- A user-provided instruction describing a manipulation goal and an image of the scene.\n"
            "Your task is to:\n"
            "- Generate a rephrased command for the robot.\n"
        ),
        "user": (
            "Task: {instruction}.\n"
            "Generate a command for the robot to successfully complete the task.\n"
        ),
    },
}

THINKING_SUFFIX = (
    "\nThink inside <think></think> tags following these steps:\n"
    "1. Identify the object to pick and the placement target in the scene.\n"
    "2. Reason about how the robot likely perceives each.\n"
    "3. Decide how to rephrase the command.\n"
    "Keep thinking process short, no more than 10 sentences."
    "Then give your final command inside <answer> your_command_here </answer> tags.\n"
    "NOTE: only the command inside <answer></answer> will be passed to VLA!"
)

THINKING_SUFFIX_SIMPLE = (
    "\n"
    "Think inside <think></think> tags:\n"
    "- Look at the image and the task.\n"
    "- Decide how to phrase the command.\n"
    "Keep thinking concise.\n"
    "Then give your final command inside <answer>your_command</answer> tags.\n"
    "Only the text inside <answer></answer> is sent to the robot."
)

SIMPLE_SUFFIX = "Look at the image, think about the scene, then write the command.\n"


def build_messages(
    instruction: str,
    image: Image.Image,
    prompt_key: str = "guided",
    thinking: bool = False,
) -> list[dict]:
    try:
        prompt = PROMPTS[prompt_key]
    except KeyError as error:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"Unknown prompt {prompt_key!r}; available prompts: {available}") from error
    if thinking:
        suffix = THINKING_SUFFIX_SIMPLE if prompt_key == "simple" else THINKING_SUFFIX
    else:
        suffix = SIMPLE_SUFFIX
    user = prompt["user"] + suffix
    return [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user.format(instruction=instruction)},
            ],
        },
    ]
