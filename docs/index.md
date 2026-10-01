---
title: E3SM AI Group
description: Energy Exascale Earth System Modeling AI Group
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

<div class="hero-logo">
  <img src="assets/images/E3SM_Logo.png" alt="E3SM Logo" draggable="false">
</div>

# AI Group

<div class="hero-actions">
  <a href="#introduction" class="md-button md-button--primary md-button--pill">Get Started</a>
</div>

</div>

## Introduction

This space serves as a collective working space for E3SM AI Group.
We share docs, scripts, examples, and prototypes for E3SM AI efforts.

## Relationship to AI2 ACE / FME

Most of the workflows here use the [ACE (AI2 Climate Emulator)](https://ai2-climate-emulator.readthedocs.io/)
model and its training/inference framework, [FME](https://ai2-climate-emulator.readthedocs.io/),
both developed by [AI2](https://allenai.org/). We do **not** re-document ACE or FME here. For
authoritative information on configuration options, model architecture, and APIs, please refer
to the upstream AI2 documentation:

- **ACE / FME documentation**: <https://ai2-climate-emulator.readthedocs.io/>
- **ACE training configuration**: <https://ai2-climate-emulator.readthedocs.io/en/latest/training_config.html>
- **AI2 model weights / datasets on Hugging Face**: <https://huggingface.co/allenai>

This site collects the E3SM-specific glue: how to set up environments on E3SM-relevant HPC systems
(e.g., NERSC Perlmutter), how to produce ACE-ready training data from E3SM (EAMxx) runs, and how
to run inference with E3SM-flavored checkpoints. The code lives in our fork,
[E3SM-Project/ace](https://github.com/E3SM-Project/ace), which tracks AI2's upstream.

## Getting Started

To access our quick guides and examples, click on the **Quick Guides** tab at the top of the page.