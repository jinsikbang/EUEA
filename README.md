<div align="center">

# EUEA: Environmental Understanding Vision-Language Model for Embodied Agent

<p>
<a href="https://arxiv.org/abs/2604.19839"><img src="https://img.shields.io/badge/arXiv-2604.19839-b31b1b.svg?style=flat" alt="arXiv"></a>
<a href="https://eu-ea.github.io/"><img src="https://img.shields.io/badge/Project-Page-1f72ff.svg?style=flat" alt="Project Page"></a>
<a href="https://huggingface.co/bangskitchen"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Data-yellow.svg?style=flat" alt="Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat" alt="License: MIT"></a>
</p>

**[Jinsik Bang](https://eu-ea.github.io/), Jaeyeon Bae, Donggyu Lee, Siyeol Jung, Taehwan Kim**

Ulsan National Institute of Science and Technology (UNIST)

_CVPR 2026_

</div>

---

## Overview

Vision-language models (VLMs) are increasingly used as the reasoning backbone of
embodied agents, yet they still struggle to *understand* the environment while a
task is being carried out. **EUEA** is a framework that strengthens the
environmental understanding of VLMs by fine-tuning four core competencies that an
embodied agent needs throughout an episode:

- **🔎 Object Perception** — identifying the objects that are relevant to the current task.
- **🗺️ Task Planning** — generating the sequence of interaction subgoals to reach the goal.
- **🤖 Action Understanding** — judging the likelihood that a chosen action will succeed.
- **🎯 Goal Recognition** — determining whether the (sub)goal has been completed.

On top of these skills, EUEA adds a **recovery** step that samples alternative
actions to correct failure cases, and a **Group Relative Policy Optimization
(GRPO)** stage that refines inconsistent predictions. Together these yield an
**+8.86% absolute improvement in success rate on ALFRED**, with further gains from
the recovery and optimization stages. Our skill-level analysis also reveals where
current VLMs fall short in environmental understanding.

<div align="center">
  <a href="https://eu-ea.github.io/">📄 Project Page</a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2604.19839">📝 Paper</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/bangskitchen">🤗 Models &amp; Datasets</a>
</div>

## News

- **2026** — EUEA is accepted to **CVPR 2026** 🎉
- **2026** — Paper, project page, datasets, and model checkpoints are released.

## Resources

| Resource | Link |
| --- | --- |
| 📝 Paper (arXiv) | https://arxiv.org/abs/2604.19839 |
| 🌐 Project Page | https://eu-ea.github.io/ |
| 🤗 Datasets (`EUEA-LangR`) | https://huggingface.co/bangskitchen — *EUEA-LangR* collection |
| 🤗 Checkpoints (`EUEA-ALFRED`) | https://huggingface.co/bangskitchen — *EUEA-ALFRED* collection |

## Datasets

EUEA is built and evaluated on two embodied benchmarks, each with seen and unseen
splits for training and validation:

- **ALFRED** — instruction-following household tasks in interactive 3D scenes.
- **LangR** — language-grounded rearrangement tasks.

Skill-level supervision is provided for each competency (object grounding, object
interaction, planning, action grounding, action anticipation, and goal
recognition), enabling both fine-tuning and fine-grained evaluation.

All datasets are released in the
[**EUEA-LangR**](https://huggingface.co/bangskitchen) collection on Hugging Face.

## Model Checkpoints

Fine-tuned EUEA checkpoints are released in the
[**EUEA-ALFRED**](https://huggingface.co/bangskitchen) collection on Hugging Face.
The framework is model-agnostic and has been validated on a range of open
vision-language backbones, including:

- InternVL2.5-4B / InternVL3-8B
- Qwen2.5-VL-3B
- MiniCPM-V-4.5

## Getting Started

> 🚧 **Code coming soon.** Training and evaluation code will be released here. (2026/6/5)
> In the meantime, the datasets and model checkpoints are already available on
> [Hugging Face](https://huggingface.co/bangskitchen). Star ⭐ / watch 👀 this repo
> for updates.

## Citation

If you find EUEA useful in your research, please consider citing:

```bibtex
@misc{bang2026euea,
  title         = {Environmental Understanding Vision-Language Model for Embodied Agent},
  author        = {Jinsik Bang and Jaeyeon Bae and Donggyu Lee and Siyeol Jung and Taehwan Kim},
  year          = {2026},
  eprint        = {2604.19839},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.19839}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

We thank the authors of [ALFRED](https://askforalfred.com/), [LangR](https://github.com/apple/ml-llarp) and the open vision-language model community ([InternVL](https://github.com/OpenGVLab/InternVL),
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)) whose work made this research
possible.