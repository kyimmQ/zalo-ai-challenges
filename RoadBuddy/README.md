# RoadBuddy – Understanding the Road through Dashcam AI

## Description

Traffic is a critical issue in today's society. The challenge "RoadBuddy – Understanding the Road through Dashcam AI" aims to build a driving assistant capable of understanding video content from dashcams to quickly answer questions about traffic signs, signals, and driving instructions. This helps enhance safety, ensure legal compliance, and reduce driver distraction. The solution is also useful for post-accident analysis, evidence retrieval, logistics optimization, and enriching map/infrastructure data from common camera sources.

From a research perspective, the challenge creates a Vietnamese benchmark tailored to the traffic context in Vietnam.

## Task

### Input

- A traffic video recorded by a car dashcam mounted on a car, lasting from 5 to 15 seconds in various scenarios such as:
  - Urban/highway traffic
  - Day/night conditions
  - Rain/sun weather
  - May include traffic signs, signals, lane arrows, road markings, vehicles, etc.
- A user question

### Output

Corresponding answer to the question.

**Note:** Knowledge used to answer questions must comply with current Vietnamese road traffic laws.

## Evaluation

### Scoring Methodology

- Each test case is run through the system to get results
- Maximum runtime per test case: **30 seconds**
- Each submitted answer is scored as:
  - **Correct (1 point)** if it matches the ground truth
  - **Incorrect (0 points)** otherwise
- Answers that exceed the 30-second time limit receive **0 points**
- System score is calculated based on the total number of correct answers divided by the total number of test cases

### Formula

```
Accuracy = Number of correct answers / Total number of test cases
```

### Tie-breaker

When teams have equal accuracy scores, final ranking is determined by average inference time (lower is better).

## Data

### Dataset Overview

- **Training data:** ~600 videos, ~1000 samples including: questions, videos, answers, support frames
- **Public test:** ~300 videos, ~500 samples including: questions, videos
- **Private test:** ~300 videos, ~500 samples including: questions, videos

### Training Data Format

Training dataset includes a `train.json` file and a folder of traffic videos. Each item in the annotations includes:

- `video_path`: the video which is used for the question
- `question`: the question
- `choices`: choices of the questions
- `answer`: the correct answer
- `support_frames`: reference frame in videos at a specific time

The provided public test and private test datasets have the same format, but answer and supported frames are not provided.

### Download

- Training data and Public test: [download](https://dl-challenge.zalo.ai/2025/TrafficBuddyHZUTAXF/traffic_buddy_train+public_test.zip)

## Submission Format

Each team builds a solution for answering each question in the public and private test, then submits to the submission page of the challenge.

The format of submission file is a **CSV format** as follows:

```csv
question_id,answer
1,A
2,B
3,C
...
```

## Rules

- **Model size** at inference time ≤ **8B parameters**. You can use several small models if needed.
- **Inference time** ≤ **30s/sample**
- **Target hardware:** The machine for running the Docker of the final solution is configured with:
  - 1 GPU (RTX 3090 or NVIDIA A30)
  - CPU: 16 cores
  - RAM: 64GB
  - Intel(R) Xeon(R) Gold 6442Y
- **No Internet access** during inference
- **Open-source data/models allowed**
- **Synthetic data generation allowed** using services or other models (LLM, VLM, etc.)
- After the competition ends, participants commit not to store any training data for personal purposes

## Getting Started

1. Download the training data and public test
2. Explore the dataset structure
3. Develop your model (≤8B parameters)
4. Ensure inference time ≤30s per sample
5. Generate predictions in CSV format
6. Submit to the challenge platform

## Notes

- Focus on understanding Vietnamese traffic context
- Comply with current Vietnamese road traffic laws
- Optimize for both accuracy and inference speed
- Consider multi-modal approaches (video + text)
