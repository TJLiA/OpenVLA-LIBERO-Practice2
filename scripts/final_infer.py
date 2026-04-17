import os

import imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


MODEL_PATH = "/root/autodl-tmp/models/vla-libero-full"
LIBERO_ROOT = "/root/autodl-tmp/LIBERO"
OUTPUT_VIDEO = "/root/autodl-tmp/vla_result.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 100
FPS = 20


def run_libero_inference_and_save_video() -> None:
    print("Loading model and processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).to(DEVICE)
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        return

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(0)
    task_description = task.language

    bddl_path = os.path.join(
        LIBERO_ROOT,
        "libero/libero/bddl_files",
        task.problem_folder,
        task.bddl_file,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=128,
        camera_widths=128,
    )

    obs = env.reset()
    print(f"Environment ready. Task: {task_description}")

    frames = []
    try:
        for step in range(MAX_STEPS):
            image_raw = np.flipud(obs["agentview_image"])
            frames.append(image_raw.astype("uint8"))

            image_pil = Image.fromarray(image_raw.astype("uint8"))
            inputs = processor(task_description, image_pil, return_tensors="pt").to(
                DEVICE,
                dtype=torch.bfloat16,
            )

            with torch.no_grad():
                action = model.predict_action(**inputs, unnorm_key="libero_10")

            obs, reward, done, info = env.step(action)

            if step % 20 == 0:
                print(f"Recorded {step}/{MAX_STEPS} steps")
    except Exception as exc:
        print(f"Inference interrupted: {exc}")

    print("Writing video...")
    imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS)
    print(f"Saved video to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    run_libero_inference_and_save_video()
