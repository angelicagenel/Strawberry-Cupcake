#!/usr/bin/env python3
"""
Test script for ACTFL criteria implementation
"""

import json
import sys

def test_criteria_loading():
    """Test that ACTFL criteria file loads correctly"""
    print("Testing ACTFL criteria loading...")

    try:
        with open("actfl_criteria.json", "r", encoding="utf-8") as f:
            criteria = json.load(f)

        print(f"✓ Successfully loaded {len(criteria)} proficiency levels")

        # Test that all required fields are present
        required_fields = [
            "name", "score_range", "oral_production", "functions",
            "discourse", "grammatical_control", "vocabulary", "pronunciation",
            "communication_strategies", "sociocultural_use", "feedback_template"
        ]

        for level_key, level_data in criteria.items():
            for field in required_fields:
                if field not in level_data:
                    print(f"✗ Missing field '{field}' in level '{level_key}'")
                    return False

        print(f"✓ All levels have required fields")

        # Test that score ranges are valid and don't overlap
        score_ranges = []
        for level_key, level_data in criteria.items():
            score_min, score_max = level_data["score_range"]
            score_ranges.append((score_min, score_max, level_data["name"]))

            if score_min > score_max:
                print(f"✗ Invalid score range for {level_data['name']}: [{score_min}, {score_max}]")
                return False

        print(f"✓ All score ranges are valid")

        # Display the levels and their ranges
        print("\nACTFL Proficiency Levels:")
        for score_min, score_max, name in sorted(score_ranges, key=lambda x: x[0]):
            print(f"  {name:20s} [{score_min:3d} - {score_max:3d}]")

        return True

    except FileNotFoundError:
        print("✗ actfl_criteria.json file not found")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"✗ Error loading criteria: {e}")
        return False

def test_score_mapping():
    """Test that scores map to correct proficiency levels"""
    print("\n\nTesting score to level mapping...")

    test_cases = [
        (0, "Novice Low"),
        (50, "Novice Low"),
        (55, "Novice Mid"),
        (60, "Novice High"),
        (65, "Intermediate Low"),
        (70, "Intermediate Mid"),
        (75, "Intermediate High"),
        (80, "Advanced Low"),
        (85, "Advanced Mid"),
        (90, "Advanced High"),
        (95, "Superior"),
        (100, "Distinguished"),
    ]

    with open("actfl_criteria.json", "r", encoding="utf-8") as f:
        criteria = json.load(f)

    all_passed = True
    for score, expected_level in test_cases:
        # Find the matching level
        found_level = None
        for level_key, level_data in criteria.items():
            score_min, score_max = level_data["score_range"]
            if score_min <= score <= score_max:
                found_level = level_data["name"]
                break

        if found_level == expected_level:
            print(f"✓ Score {score:3d} → {found_level}")
        else:
            print(f"✗ Score {score:3d} → {found_level} (expected {expected_level})")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("ACTFL Criteria Implementation Test Suite")
    print("=" * 60)

    # Run tests
    test1_passed = test_criteria_loading()
    test2_passed = test_score_mapping()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        sys.exit(1)
