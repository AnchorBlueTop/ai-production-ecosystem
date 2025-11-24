<h1 align="center">AI Creative Production: An Engineering Case Study</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2005.05535">
    <img src="https://static.arxiv.org/static/browse/0.3.0/images/icons/favicon.ico" width=14></img>
    This project is built upon the foundational framework of DeepFaceLab, detailed in arXiv:2005.05535.
  </a>
</p>

---

### 1. Executive Summary

This repository is a comprehensive engineering case study of the custom ecosystem I built for producing professional-grade AI digital art. My creative work, which involves self-insertion deepfakes into iconic Bollywood scenes for my **[@wholeface_](https://www.instagram.com/wholeface_)** portfolio (10k+ followers), demanded a level of performance and stability that off-the-shelf tools could not provide.

To solve this, I engineered a complete, end-to-end solution spanning custom hardware, a meticulously documented production workflow, and a data-driven model training methodology. This document details the engineering decisions, technical processes, and expert knowledge developed over hundreds of hours of hands-on work.

![resized intro](https://github.com/AnchorBlueTop/Personal-Machine-Learning-Model/assets/98157644/8d3a0b5b-c235-4103-aced-795cfdf86a56)

---

### 2. The System Architecture: A No-Compromises ML Ecosystem

To enable the intensive, multi-day training sessions required for high-fidelity output, I designed and built a specialized hardware and software environment from the ground up.

#### Hardware Foundation: The ML Workstation

The system was engineered for a single purpose: maximum, sustained performance and thermal stability for 24/7 operation.

| Component | Specification | Engineering Purpose |
|---|---|---|
| **CPU** | AMD Ryzen 9 9950X3D (Delidded)| AVX-512 support for accelerated CPU-bound tasks like the SOT color transfer algorithm during the final merge process. |
| **GPU** | NVIDIA RTX 5090 (32GB) | Maximum available VRAM to enable training of higher-resolution models and larger batch sizes. |
| **Cooling**| Custom Water Loop w/ External MO-RA IV Radiator | To decouple heat dissipation from the workstation. The external radiator, wall-mounted in an adjacent room, allows for silent, continuous, and stable operation without thermal throttling, which is critical for multi-day training runs. |
  
<div align="center">
  <table>
    <!-- Top Row: Two Portrait Images -->
    <tr>
      <td align="center" width="50%">
        <img src="https://github.com/AnchorBlueTop/ai-production-ecosystem/blob/main/assets/pc-on.jpg?raw=true" alt="The complete ML Workstation, powered on." width="400">
        <br/>
        <b>The Complete ML Workstation</b>
      </td>
      <td align="center" width="50%">
        <img src="https://github.com/AnchorBlueTop/ai-production-ecosystem/blob/main/assets/pc-side.jpg?raw=true" alt="Custom 3D-printed pump and sensor mount." width="400">
        <br/>
        <b>Custom 3D-Printed Sensor Mounts</b>
      </td>
    </tr>
    <!-- Middle Row: One Landscape Image -->
    <tr>
      <td align="center" colspan="2">
        <img src="https://github.com/AnchorBlueTop/ai-production-ecosystem/blob/main/assets/external-tubing-from-pc.jpg?raw=true" alt="Clean external tube routing." width="810">
        <br/>
        <b>Clean & Organized External Tube Routing</b>
      </td>
    </tr>
    <!-- Bottom Row: One Landscape Image -->
    <tr>
      <td align="center" colspan="2">
        <img src="https://raw.githubusercontent.com/AnchorBlueTop/ai-production-ecosystem/refs/heads/main/assets/external-mora.jpg" alt="The external MO-RA IV radiator." width="810">
        <br/>
        <b>External Radiator for 24/7 Thermal Stability</b>
      </td>
    </tr>
  </table>
</div>

    
---

### 3. Engineering Contributions: Workflow & Performance Enhancements

My deep involvement with this workflow naturally led me to identify and solve critical inefficiencies and bugs within the original DeepFaceLab software. I have moved beyond being a user to actively engineering improvements to the toolset. **The modified source code for these contributions can be reviewed in the [`/src`](/src) directory.**

#### Bi-Directional Masking: A Quality-of-Life Overhaul
*   **Problem:** The default mask control was a single, uniform modifier. This made it impossible to create precise masks for complex head shapes, often resulting in background bleed or an unnatural "helmet" look.
*   **Solution:** I implemented a complete overhaul of the mask modification system, engineering a new bi-directional control feature. This allows for independent, per-axis expansion and erosion of the mask (top, bottom, left, and right). **View the implementation in [`src/MergeMasked.py`](/src/MergeMasked.py)**.
*   **Impact:** This enhancement transforms a frustrating, limited tool into a powerful, precise instrument. It enables the creation of much cleaner, more realistic masks that can be tailored to fit any subject perfectly. The difference between the old and new systems is demonstrated below.

<div align="center">
  <table>
    <tr>
      <td align="center" valign="top">
        <img src="https://raw.githubusercontent.com/AnchorBlueTop/ai-production-ecosystem/refs/heads/main/assets/fixed-directional-mask.png" alt="Old fixed mask expansion" width="400">
        <br/>
        <b>Before: Uniform Expansion</b>
        <br/>
        <i>A single value controls all four directions, offering no flexibility.</i>
      </td>
      <td align="center" valign="top">
        <img src="https://raw.githubusercontent.com/AnchorBlueTop/ai-production-ecosystem/refs/heads/main/assets/bi-directional-mask.png" alt="New flexible mask expansion" width="400">
        <br/>
        <b>After: Independent Per-Axis Control</b>
        <br/>
        <i>Separate values for top, bottom, left, and right allow for precise adjustments.</i>
      </td>
    </tr>
    <tr>
      <td align="center" colspan="2">
        <img src="https://raw.githubusercontent.com/AnchorBlueTop/ai-production-ecosystem/refs/heads/main/assets/bi-directional-mask-example.png" alt="Example of directional mask expansion in use" width="810">
        <br/>
        <b>Result: A precise mask that covers the forehead and hair without distorting the sides.</b>
      </td>
    </tr>
  </table>
</div>

In addition to developing the core feature, this work included:
- **Fixing Core Logic Bugs:** I diagnosed and fixed a bug in the underlying OpenCV implementation that caused inverted and unpredictable control behavior.
- **Preserving Exclusion Zones:** I added logic to ensure that manually drawn exclusion masks (e.g., for glasses or beards) are respected during the expansion process.
- **Ergonomic Key Remapping:** I redesigned the entire control scheme for the interactive merger, creating an intuitive, clustered key layout that significantly improves workflow speed.

#### XSeg Training Pipeline: Eliminating CPU Bottlenecks
*   **Problem:** I identified a severe producer-consumer bottleneck in the XSeg training pipeline where the GPU was frequently idle ("starving") while waiting for the CPU to prepare data batches.
*   **Solution:** I refactored the data loader for the XSeg model, removing an artificial 8-core cap and implementing new logic to intelligently allocate `(Total CPU Cores - 2)` processes for data generation. **View the implementation in [`src/Model.py`](/src/Model.py)**.
*   **Impact:** On my 16-core CPU, this increased data generation parallelism by **3.5x**, completely eliminating the GPU starvation issue and leading to smoother, faster training performance.

---

### 4. The End-to-End Production Workflow (~30+ Hours Per Project)

Each video is the result of a painstaking, multi-stage production pipeline that I have refined for quality and consistency.

1.  **Data Acquisition (MP4 to PNG):** Source and destination video clips are carefully selected and decomposed into high-quality PNG image sequences.
2.  **Face Extraction & Alignment:** Faces are extracted from every frame. This is the most labor-intensive stage.
    *   **Manual XSeg Polygon Drawing:** I manually draw precise segmentation polygons around the destination face. This often requires **hours of meticulous work per video**, with complex scenes sometimes requiring over **400 individual face alignments**.
3.  **XSeg Model Training:** A dedicated XSeg segmentation model is trained on the manually labeled data to produce high-quality, context-aware masks.
4.  **SAEHD Model Training:** The core deepfake model is trained. This is a multi-day, data-driven process where I make critical decisions based on performance metrics. *(See Section 5 for a detailed breakdown).*
5.  **Merging & Compositing:** The trained model is used to merge my face onto the destination frames, followed by final color grading and video assembly.

<div align="center">
  <img src="https://raw.githubusercontent.com/AnchorBlueTop/ai-production-ecosystem/refs/heads/main/assets/xseg-masking.gif" alt="Time-lapse of the manual XSeg polygon drawing process." width="810">
  <br/>
  <b>A 3000x time-lapse of the meticulous manual XSeg polygon drawing process.</b>
  <br/>
  <i>This critical step ensures a high-quality, precise mask for the final output.</i>
</div>

---

### 5. The Art of Model Training: An Engineering Methodology

Training is not a "fire-and-forget" process. It is an iterative cycle of analysis and intervention, where I make informed, data-driven decisions based on performance metrics to guide the model toward an optimal result.

#### How to Make Informed Decisions During Training
Knowing when to proceed to the next phase comes down to looking at the face previews and interpreting model loss values. The model loss usually follows a logarithmic trend downwards during the initial generalization phase before stagnating during the refinement and final sharpening phases.

<div align="center">
  <img src="https://github.com/AnchorBlueTop/Personal-Machine-Learning-Model/assets/98157644/341e3a21-cd55-4cda-960c-3043a56717f4" alt="Loss Curve Graph">
  <p>The entire training process is a balancing act between two key metrics, which are logged continuously:</p>
  <img src="https://github.com/AnchorBlueTop/Personal-Machine-Learning-Model/assets/98157644/5899a485-cdb6-47b1-bed6-be0a89458dfb" alt="Loss Value Log">
  <br/>
</div>

*   **SRC (Source) Loss:** This measures how accurately the model can reconstruct my own face. A lower value means the model is getting better at reproducing my specific facial features.
*   **DST (Destination) Loss:** This measures how accurately the model can reconstruct the face of the actor in the target video. A lower value indicates better performance in capturing the target's expressions and head movements.

Essentially, once the loss values no longer improve at a significant rate (I typically use a threshold of less than **0.010 improvement over a 25-minute interval**), and there isn't a notable difference in the visual previews, it's time to move to the next step or phase. This data-driven approach is critical to avoid overfitting and achieve a high-quality result.

#### My Three-Phase Training Process
My methodology is broken down into three distinct phases, with progression between them governed by the data described above.

##### Phase 1: Generalization (Random Warp Enabled)
*   **Objective:** To teach the model the general structure, expressions, and angles of the faces.
*   **Key Settings:** `Random Warp (RW)`: **Enabled**, `Flip DST faces randomly`: **Enabled**.

##### Phase 2: Refinement & Detail Acquisition (Random Warp Disabled)
*   **Objective:** To learn the fine details, textures, and subtle nuances of the face.
*   **Key Settings & Sub-stages:**
    1.  **Initial Refinement:** `RW` is **disabled**, `LRD` is **enabled**, and `MS-SSIM` loss function is introduced.
    2.  **Pose Refinement:** `Uniform Yaw (UY)` is enabled for several hours to improve side profiles.
    3.  **Feature Priority:** `Eyes and Mouth Priority (EMP)` is selectively enabled to fix specific blurriness, used with caution to avoid artifacts.

##### Phase 3: Final Sharpening using Generative Adversarial Network (GAN)
*   **Objective:** To apply a final layer of sharpness and realism.
*   **Execution:** `GAN` is enabled at a low power setting (`0.1`) for a very short duration (**10-15 minutes**) to avoid over-sharpening artifacts.

---

### 6. Model Architectures & Parameters: An Empirical Analysis

There is no single "best" model. The optimal architecture is a delicate balance of resolution, model complexity (dimensions), VRAM usage, and training time. Through hundreds of hours of empirical testing, I have found that the `LIAE-UDT` architecture provides a strong balance of source face likeness and adaptation to destination lighting.

#### Understanding the Key Parameters
My training methodology is based on a deep, practical understanding of how each parameter affects the final output.

*   **Autoencoder Dimensions (AE Dims):** This can be thought of as the model's total "knowledge capacity"—the size of its internal library for facial features.
*   **Encoder/Decoder Dimensions (E/D Dims):** If the Autoencoder is the library, the Encoder and Decoder are the "PhD-level readers and writers." Higher dimensions allow them to capture and reproduce more intricate details.
*   **Batch Size:** This is critical for training stability. Through extensive testing, I have found that a batch size below 8 often leads to poor quality outputs, such as misaligned eyes.

#### Model Configurations & Findings
Below are my most frequently used and tested model configurations, representing a history of my work and hardware progression.

##### Primary Production Models (RTX 4090)
*This library of models was developed on the RTX 4090. The 480p model, with over 4.8 million iterations, represents my most refined and stable configuration on that platform.*
<div align="center">

| Resolution | **480x480 (Main)** | 512x512 |
|---|---|---|
| **Architecture** | LIAE-UDT | DF-UD |
| **AE Dims** | 292 | 256 |
| **Encoder Dims**| 78 | 72 |
| **Decoder Dims**| 78 | 72 |
| **Mask Dims** | 32 | 18 |
| **Batch Size** | 10 | 10 |

</div>

##### High-Performance Models (RTX 5090)
*These models leverage the increased VRAM of the new hardware to push for higher resolutions and model complexity.*
<div align="center">

| Parameter | High Res (Balanced) | High Res (Testing) |
|---|---|---|
| **Resolution** | 544x544 | 544x544 |
| **AE Dims** | 394 | 402 |
| **Encoder Dims** | 80 | 98 |
| **Decoder Dims** | 80 | 98 |
| **Mask Dims**| 32 | 32 |
| **Batch Size** | 10 | 10 |

</div>

*Note on the AdaBelief Optimizer: While this optimizer can sometimes accelerate the initial learning phase, its severe VRAM penalty and workflow friction make it suboptimal for my process. For these reasons, I currently favor the standard RMSprop optimizer for its stability and reusability.*

---

#### Training Previews & Results
The following images are examples of the output quality achieved through my refined training methodology.

<div align="center">
  <img src="https://github.com/AnchorBlueTop/Personal-Machine-Learning-Model/assets/98157644/2026d10e-061a-4c12-96eb-2ca999f0be03" alt="Preview 1">
  <br/>
  <img src="https://github.com/AnchorBlueTop/Personal-Machine-Learning-Model/assets/98157644/a7bb75a2-ec37-4460-8633-c591cd110d37" alt="Preview 2">
  <br/>
  <img src="https://github.com/AnchorBlueTop/Personal-Machine-Learning-Model/assets/98157644/6a97dd3d-abf4-4327-9a08-3c6412fedadb" alt="Preview 3">
</div>

---

### 7. Conclusion: From Passion Project to Engineering Proof

What began as a creative hobby, a way to explore a unique form of digital art—quickly evolved into a holistic engineering challenge. The pursuit of professional-grade quality demanded more than just using a tool; it required me to build, optimize, and master an entire ecosystem at every level:

*   **As a Hardware Engineer,** I designed and built a specialized, thermally-stable workstation for 24/7 operation.
*   **As a Software Engineer,** I delved into a legacy codebase to identify bottlenecks, fix bugs, and implement significant quality-of-life enhancements.
*   **As a Machine Learning Practitioner,** I developed a data-driven, empirical methodology for model training, making critical decisions based on performance metrics to achieve the highest quality output.

For me, this technology has been an art form—a way to achieve a personal dream of placing myself into the iconic Bollywood movies I grew up with. While I am acutely aware of the potential for misuse of AI, I believe this project stands as a testament to its power as a creative and expressive medium. It is a demonstration of responsible, passionate, and technically meticulous application of this technology.

This project serves as a comprehensive portfolio of my practical engineering expertise. This multi-year, self-directed endeavor proves my capabilities in:

*   **Systems Thinking:** Architecting a complete hardware and software solution to solve a complex problem.
*   **Software Engineering:** Analyzing, debugging, and enhancing a production-level Python application.
*   **ML Operations (MLOps):** Managing the entire machine learning pipeline from data preparation and model training to final production and output.
*   **Hypothesis-Driven R&D:** Systematically testing solutions, documenting failures (like the FP16 investigation), and making strategic pivots based on empirical evidence.

Ultimately, this project is a testament to my passion for building, my persistence through complex challenges, and my ability to deliver a complete, high-quality product from the ground up.

---
