"""
RoadBuddy Dataset Exploration Script
This script helps you understand the dataset structure and statistics.
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_training_data(train_json_path):
    """Analyze training dataset"""
    print("=" * 70)
    print("TRAINING DATA ANALYSIS")
    print("=" * 70)
    
    data = load_json(train_json_path)
    samples = data['data']
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total count field: {data.get('__count__', 'N/A')}")
    
    # Video statistics
    video_paths = [s['video_path'] for s in samples]
    unique_videos = set(video_paths)
    print(f"\n🎥 Video Statistics:")
    print(f"  Unique videos: {len(unique_videos)}")
    print(f"  Avg questions per video: {len(samples) / len(unique_videos):.2f}")
    
    # Questions per video distribution
    video_counts = Counter(video_paths)
    max_questions = max(video_counts.values())
    print(f"  Max questions from one video: {max_questions}")
    
    # Find video with most questions
    most_used_video = video_counts.most_common(1)[0]
    print(f"  Most used video: {most_used_video[0]}")
    print(f"    → {most_used_video[1]} questions")
    
    # Choice distribution
    print(f"\n📝 Question Format:")
    choice_counts = Counter(len(s['choices']) for s in samples)
    for num_choices, count in sorted(choice_counts.items()):
        print(f"  {num_choices} choices: {count} questions ({count/len(samples)*100:.1f}%)")
    
    # Answer distribution
    print(f"\n✅ Answer Distribution:")
    answers = [s['answer'][0] for s in samples]  # Get first letter (A, B, C, D)
    answer_counts = Counter(answers)
    for answer, count in sorted(answer_counts.items()):
        print(f"  {answer}: {count} ({count/len(samples)*100:.1f}%)")
    
    # Support frames analysis
    print(f"\n🎯 Support Frames:")
    support_frame_counts = [len(s['support_frames']) for s in samples]
    avg_support_frames = sum(support_frame_counts) / len(samples)
    print(f"  Avg support frames per question: {avg_support_frames:.2f}")
    
    # Get timestamp statistics
    all_timestamps = [ts for s in samples for ts in s['support_frames']]
    if all_timestamps:
        print(f"  Min timestamp: {min(all_timestamps):.2f}s")
        print(f"  Max timestamp: {max(all_timestamps):.2f}s")
        print(f"  Avg timestamp: {sum(all_timestamps)/len(all_timestamps):.2f}s")
    
    # Question keyword analysis
    print(f"\n🔍 Common Keywords in Questions:")
    all_questions = ' '.join([s['question'] for s in samples]).lower()
    
    keywords = {
        'biển báo': all_questions.count('biển báo'),
        'đèn': all_questions.count('đèn'),
        'rẽ phải': all_questions.count('rẽ phải'),
        'rẽ trái': all_questions.count('rẽ trái'),
        'đi thẳng': all_questions.count('đi thẳng'),
        'làn': all_questions.count('làn'),
        'cấm': all_questions.count('cấm'),
        'đường': all_questions.count('đường'),
        'phương tiện': all_questions.count('phương tiện'),
    }
    
    for keyword, count in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{keyword}': {count} times")
    
    # Sample questions
    print(f"\n📋 Sample Questions:")
    for i in range(min(3, len(samples))):
        s = samples[i]
        print(f"\n  {i+1}. ID: {s['id']}")
        print(f"     Q: {s['question']}")
        print(f"     Choices: {s['choices']}")
        print(f"     Answer: {s['answer']}")
        print(f"     Support frames: {s['support_frames']}")
        print(f"     Video: {s['video_path']}")
    
    return samples


def analyze_test_data(test_json_path):
    """Analyze public test dataset"""
    print("\n" + "=" * 70)
    print("PUBLIC TEST DATA ANALYSIS")
    print("=" * 70)
    
    data = load_json(test_json_path)
    samples = data['data']
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Total count field: {data.get('__count__', 'N/A')}")
    
    # Video statistics
    video_paths = [s['video_path'] for s in samples]
    unique_videos = set(video_paths)
    print(f"\n🎥 Video Statistics:")
    print(f"  Unique videos: {len(unique_videos)}")
    print(f"  Avg questions per video: {len(samples) / len(unique_videos):.2f}")
    
    # Choice distribution
    print(f"\n📝 Question Format:")
    choice_counts = Counter(len(s['choices']) for s in samples)
    for num_choices, count in sorted(choice_counts.items()):
        print(f"  {num_choices} choices: {count} questions ({count/len(samples)*100:.1f}%)")
    
    # Sample questions
    print(f"\n📋 Sample Test Questions:")
    for i in range(min(3, len(samples))):
        s = samples[i]
        print(f"\n  {i+1}. ID: {s['id']}")
        print(f"     Q: {s['question']}")
        print(f"     Choices: {s['choices']}")
        print(f"     Video: {s['video_path']}")
    
    return samples


def check_video_files(base_dir):
    """Check if videos have been downloaded"""
    print("\n" + "=" * 70)
    print("VIDEO FILES CHECK")
    print("=" * 70)
    
    train_video_dir = Path(base_dir) / "train" / "videos"
    test_video_dir = Path(base_dir) / "public_test" / "videos"
    
    train_videos = list(train_video_dir.glob("*.mp4")) if train_video_dir.exists() else []
    test_videos = list(test_video_dir.glob("*.mp4")) if test_video_dir.exists() else []
    
    print(f"\n📁 Training videos:")
    print(f"  Directory: {train_video_dir}")
    print(f"  Videos found: {len(train_videos)}")
    if not train_videos:
        print(f"  ⚠️  No videos found! Run download_videos.py first.")
    
    print(f"\n📁 Public test videos:")
    print(f"  Directory: {test_video_dir}")
    print(f"  Videos found: {len(test_videos)}")
    if not test_videos:
        print(f"  ⚠️  No videos found! Run download_videos.py first.")
    
    return len(train_videos) > 0 and len(test_videos) > 0


def analyze_video_reuse(samples):
    """Analyze how videos are reused across questions"""
    print("\n" + "=" * 70)
    print("VIDEO REUSE ANALYSIS")
    print("=" * 70)
    
    video_to_questions = defaultdict(list)
    for s in samples:
        video_to_questions[s['video_path']].append(s['id'])
    
    # Distribution of questions per video
    questions_per_video = [len(qs) for qs in video_to_questions.values()]
    reuse_counts = Counter(questions_per_video)
    
    print(f"\n📈 Questions per Video Distribution:")
    for num_questions, num_videos in sorted(reuse_counts.items()):
        print(f"  {num_questions} question(s): {num_videos} videos")
    
    # Videos with most questions
    top_videos = sorted(video_to_questions.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print(f"\n🏆 Top 5 Most Used Videos:")
    for i, (video, questions) in enumerate(top_videos, 1):
        print(f"  {i}. {video}")
        print(f"     → {len(questions)} questions: {', '.join(questions[:3])}{' ...' if len(questions) > 3 else ''}")


def main():
    """Main execution"""
    base_dir = Path(__file__).parent / "traffic_buddy_train+public_test"
    
    # Check if dataset exists
    if not base_dir.exists():
        print(f"❌ Dataset directory not found: {base_dir}")
        print(f"   Please ensure you've extracted the dataset.")
        return
    
    train_json = base_dir / "train" / "train.json"
    test_json = base_dir / "public_test" / "public_test.json"
    
    # Analyze training data
    if train_json.exists():
        train_samples = analyze_training_data(train_json)
        analyze_video_reuse(train_samples)
    else:
        print(f"❌ Training JSON not found: {train_json}")
    
    # Analyze test data
    if test_json.exists():
        test_samples = analyze_test_data(test_json)
    else:
        print(f"❌ Test JSON not found: {test_json}")
    
    # Check video files
    check_video_files(base_dir)
    
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print("""
1. If videos are not downloaded:
   cd traffic_buddy_train+public_test
   python download_videos.py

2. Explore some sample videos to understand the content

3. Start building your solution:
   - Choose a Video-Language Model (VLM)
   - Implement frame extraction
   - Create inference pipeline
   - Generate predictions

4. For more guidance, read DATA_GUIDE.md
""")


if __name__ == "__main__":
    main()

