Here is the implementation of the function that finds the longest substring without repeating characters:
```python
from typing import Dict, List, Set
import heapq

def longest_substring_without_repeating_characters(s: str) -> int:
    """
    Returns the length of the longest substring without repeating characters.

    Args:
        s (str): The input string.

    Returns:
        int: The length of the longest substring without repeating characters.
    """
    char_set = set()  # Set to store unique characters in the current window
    left = 0  # Left pointer of the sliding window

    max_length = 0  # Maximum length found so far

    for right, char in enumerate(s):  # Iterate over the string with the right pointer
        while s[right] in char_set:  # If the character is already in the set
            char_set.remove(s[left])  # Remove the leftmost character from the set
            left += 1  # Move the left pointer to the right

        char_set.add(char)  # Add the current character to the set
        max_length = max(max_length, right - left + 1)  # Update the maximum length

    return max_length


# Example usage:
s = "abcabcbb"  # Input string with repeating characters
result = longest_substring_without_repeating_characters(s)
print(f"Length of the longest substring without repeating characters: {result}")
```
This implementation uses a sliding window approach with two pointers (`left` and `right`) to keep track of the current substring. It maintains a set `char_set` to store unique characters in the current window. When a repeating character is found, it removes the leftmost character from the set and moves the left pointer to the right. The maximum length of the substring without repeating characters is updated at each step.

The time complexity of this implementation is O(n), where n is the length of the input string, as we process each character only once.

Note that the `pydantic` library is not used in this implementation, but it can be used to validate request and response data. If you need to use Pydantic models for validation, you would need to modify the function accordingly.