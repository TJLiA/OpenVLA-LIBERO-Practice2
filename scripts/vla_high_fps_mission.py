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
VIDEO_SAVE_PATH = "/root/autodl-tmp/vla_high_fps.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 600
TARGET_FPS = 30


def run_high_fps_task() -> None:
    print(
        f"Starting high-fps recording, max_steps={MAX_STEPS}, fps={TARGET_FPS}"
    )

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
        camera_heights=512,
        camera_widths=512,
    )

    obs = env.reset()
    task_description = task.language
    print(f"Task: {task_description}")

    frames = []
    try:
        for step in range(MAX_STEPS):
            img_raw = obs["agentview_image"]
            img_corrected = np.flipud(img_raw)
            frames.append(img_corrected.astype("uint8"))

            img_pil = Image.fromarray(img_corrected.astype("uint8"))
            inputs = processor(task_description, img_pil, return_tensors="pt").to(
                DEVICE,
                dtype=torch.bfloat16,
            )

            with torch.no_grad():
                action = model.predict_action(**inputs, unnorm_key="libero_10")

            obs, reward, done, info = env.step(action)

            if step % 50 == 0:
                print(f"Ran {step}/{MAX_STEPS} steps")

            if done:
                print("Task done signal received.")
                for _ in range(20):
                    frames.append(img_corrected.astype("uint8"))
                break
    except Exception as exc:
        print(f"Run interrupted: {exc}")

    print("Writing video...")
    imageio.mimsave(VIDEO_SAVE_PATH, frames, fps=TARGET_FPS)
    print(f"Saved video to: {VIDEO_SAVE_PATH}")


if __name__ == "__main__":
    run_high_fps_task()
