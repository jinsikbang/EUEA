import os
import json
import torch
from omegaconf import DictConfig

from .base_dataset import BaseEmbodiedDataset


class SQA3DDataset(BaseEmbodiedDataset):
    """Dataset wrapper for SQA3D (Ma et al., 2023).

    SQA3D is a situated question-answering dataset in 3D scenes.  Each
    sample includes a situated description (where the agent is standing and
    what it is doing), a question, and one or more answers.

    Dataset structure expected under ``cfg.data_root``::

        sqa3d/
        ├── sqa_task/
        │   ├── balanced/
        │   │   ├── v1_balanced_sqa_annotations_train_scannetv2.json
        │   │   ├── v1_balanced_sqa_annotations_val_scannetv2.json
        │   │   └── v1_balanced_sqa_annotations_test_scannetv2.json
        │   └── questions/
        │       ├── v1_balanced_questions_train_scannetv2.json
        │       ├── v1_balanced_questions_val_scannetv2.json
        │       └── v1_balanced_questions_test_scannetv2.json
        └── scans/
            └── <scene_id>/
                ├── color/
                │   └── <frame_id>.jpg
                └── <scene_id>_pointcloud.npy
    """

    QUESTION_FILES = {
        "train": "v1_balanced_questions_train_scannetv2.json",
        "val": "v1_balanced_questions_val_scannetv2.json",
        "test": "v1_balanced_questions_test_scannetv2.json",
    }
    ANNOTATION_FILES = {
        "train": "v1_balanced_sqa_annotations_train_scannetv2.json",
        "val": "v1_balanced_sqa_annotations_val_scannetv2.json",
        "test": "v1_balanced_sqa_annotations_test_scannetv2.json",
    }

    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__(cfg, split)

    def _load_annotations(self) -> list:
        questions_path = os.path.join(
            self.data_root, "sqa3d", "sqa_task", "questions",
            self.QUESTION_FILES[self.split],
        )
        ann_path = os.path.join(
            self.data_root, "sqa3d", "sqa_task", "balanced",
            self.ANNOTATION_FILES[self.split],
        )

        with open(questions_path) as f:
            questions_data = json.load(f)
        with open(ann_path) as f:
            ann_data = json.load(f)

        # Build a mapping from question ID to answers
        answer_map: dict[str, list[str]] = {}
        for ann in ann_data.get("annotations", []):
            qid = str(ann["question_id"])
            answer_map[qid] = [a["answer"] for a in ann.get("answers", [])]

        samples = []
        for q in questions_data.get("questions", []):
            qid = str(q["question_id"])
            samples.append({
                "scene_id": q["scene_id"],
                "question_id": qid,
                "situation": q.get("situation", ""),
                "question": q["question"],
                "answers": answer_map.get(qid, []),
            })
        return samples

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        scene_id = ann["scene_id"]

        # Load a representative frame image
        images_dir = os.path.join(
            self.data_root, "sqa3d", "scans", scene_id, "color"
        )
        frame_files = sorted(os.listdir(images_dir))
        frame_path = os.path.join(images_dir, frame_files[len(frame_files) // 2])
        image = self._load_image(frame_path)

        # Load point cloud
        pc_path = os.path.join(
            self.data_root, "sqa3d", "scans", scene_id,
            f"{scene_id}_pointcloud.npy",
        )
        point_cloud = self._load_pointcloud(
            pc_path, self.cfg.get("num_points", 4096)
        )

        # Build the full prompt including situation context
        situation = ann.get("situation", "")
        question = ann["question"]
        if situation:
            full_question = f"Situation: {situation}\nQuestion: {question}"
        else:
            full_question = question

        answers = ann.get("answers", [])

        return {
            "image": image,
            "point_cloud": point_cloud,
            "scene_id": scene_id,
            "question_id": ann["question_id"],
            "situation": situation,
            "question": full_question,
            "answers": answers,
            "answer": answers[0] if answers else "",
        }
