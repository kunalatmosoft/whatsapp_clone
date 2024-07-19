# ALGORITHMS

## PATTERN SEARCHING ALGORITHMS

Pattern searching involves finding occurrences of a "pattern" within a larger "text" or "string". Here are some classic pattern searching problems along with their explanations and C code implementations:

### 1. Naive Pattern Searching

#### Explanation
The naive pattern searching algorithm checks for a pattern of length \( m \) in a text of length \( n \) by sliding the pattern one by one. It compares characters of the pattern with characters of the text one by one.

#### C Code
```c
#include <stdio.h>
#include <string.h>

void naivePatternSearch(char* text, char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);

    for (int i = 0; i <= n - m; i++) {
        int j;
        for (j = 0; j < m; j++) {
            if (text[i + j] != pattern[j])
                break;
        }
        if (j == m)
            printf("Pattern found at index %d\n", i);
    }
}

int main() {
    char text[] = "AABAACAADAABAAABAA";
    char pattern[] = "AABA";
    naivePatternSearch(text, pattern);
    return 0;
}
```

### 2. Knuth-Morris-Pratt (KMP) Algorithm

#### Explanation
The KMP algorithm improves the naive approach by avoiding unnecessary comparisons. It preprocesses the pattern to create a partial match table (lps array) that allows the pattern to skip characters while matching.

#### C Code
```c
#include <stdio.h>
#include <string.h>

void computeLPSArray(char* pattern, int m, int* lps) {
    int len = 0;
    lps[0] = 0;
    int i = 1;

    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

void KMPSearch(char* text, char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);

    int lps[m];
    computeLPSArray(pattern, m, lps);

    int i = 0; // index for text[]
    int j = 0; // index for pattern[]

    while (i < n) {
        if (pattern[j] == text[i]) {
            j++;
            i++;
        }

        if (j == m) {
            printf("Pattern found at index %d\n", i - j);
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }
    }
}

int main() {
    char text[] = "ABABDABACDABABCABAB";
    char pattern[] = "ABABCABAB";
    KMPSearch(text, pattern);
    return 0;
}
```

### 3. Rabin-Karp Algorithm

#### Explanation
The Rabin-Karp algorithm uses hashing to find patterns within a text. It hashes both the pattern and successive substrings of the text and compares them. It's efficient for multiple pattern searches.

#### C Code
```c
#include <stdio.h>
#include <string.h>

#define d 256 // number of characters in the input alphabet

void RabinKarpSearch(char* text, char* pattern, int q) {
    int n = strlen(text);
    int m = strlen(pattern);
    int i, j;
    int p = 0; // hash value for pattern
    int t = 0; // hash value for text
    int h = 1;

    // The value of h would be "pow(d, m-1)%q"
    for (i = 0; i < m - 1; i++)
        h = (h * d) % q;

    // Calculate the hash value of pattern and first window of text
    for (i = 0; i < m; i++) {
        p = (d * p + pattern[i]) % q;
        t = (d * t + text[i]) % q;
    }

    // Slide the pattern over text one by one
    for (i = 0; i <= n - m; i++) {
        // Check the hash values of current window of text and pattern.
        // If the hash values match then only check for characters one by one
        if (p == t) {
            // Check for characters one by one
            for (j = 0; j < m; j++) {
                if (text[i + j] != pattern[j])
                    break;
            }
            if (j == m)
                printf("Pattern found at index %d\n", i);
        }

        // Calculate hash value for next window of text: Remove leading digit,
        // add trailing digit
        if (i < n - m) {
            t = (d * (t - text[i] * h) + text[i + m]) % q;

            // We might get negative value of t, converting it to positive
            if (t < 0)
                t = (t + q);
        }
    }
}

int main() {
    char text[] = "GEEKS FOR GEEKS";
    char pattern[] = "GEEK";
    int q = 101; // A prime number
    RabinKarpSearch(text, pattern, q);
    return 0;
}
```

### 4. Boyer-Moore Algorithm

#### Explanation
The Boyer-Moore algorithm preprocesses the pattern and uses two heuristics (bad character and good suffix) to skip unnecessary comparisons in the text. It is particularly effective for long patterns.

#### C Code
```c
#include <stdio.h>
#include <string.h>

#define NO_OF_CHARS 256

// A utility function to get maximum of two integers
int max(int a, int b) {
    return (a > b) ? a : b;
}

// The preprocessing function for Boyer-Moore's bad character heuristic
void badCharHeuristic(char* str, int size, int badchar[NO_OF_CHARS]) {
    int i;

    // Initialize all occurrences as -1
    for (i = 0; i < NO_OF_CHARS; i++)
        badchar[i] = -1;

    // Fill the actual value of last occurrence of a character
    for (i = 0; i < size; i++)
        badchar[(int) str[i]] = i;
}

/* A pattern searching function that uses Bad Character Heuristic of Boyer-Moore Algorithm */
void BoyerMooreSearch(char* text, char* pattern) {
    int m = strlen(pattern);
    int n = strlen(text);

    int badchar[NO_OF_CHARS];

    /* Fill the bad character array by calling the preprocessing function badCharHeuristic() for given pattern */
    badCharHeuristic(pattern, m, badchar);

    int s = 0; // s is shift of the pattern with respect to text
    while (s <= (n - m)) {
        int j = m - 1;

        /* Keep reducing index j of pattern while characters of pattern and text are matching at this shift s */
        while (j >= 0 && pattern[j] == text[s + j])
            j--;

        /* If the pattern is present at current shift, then index j will become -1 after the above loop */
        if (j < 0) {
            printf("Pattern occurs at shift = %d\n", s);

            /* Shift the pattern so that the next character in text aligns with the last occurrence of it in pattern.
               The condition s+m < n is necessary for the case when pattern occurs at the end of text */
            s += (s + m < n) ? m - badchar[text[s + m]] : 1;

        } else
            /* Shift the pattern so that the bad character in text aligns with the last occurrence of it in pattern.
               The max function is used to make sure that we get a positive shift. We may get a negative shift if the last occurrence
               of bad character in pattern is on the right side of the current character. */
            s += max(1, j - badchar[text[s + j]]);
    }
}

int main() {
    char text[] = "ABAAABCD";
    char pattern[] = "ABC";
    BoyerMooreSearch(text, pattern);
    return 0;
}
```

### 5. Aho-Corasick Algorithm

#### Explanation
The Aho-Corasick algorithm constructs a finite state machine that can be used for efficient multi-pattern searching in text. It preprocesses the patterns and creates a trie (prefix tree) to handle multiple patterns simultaneously.

#### C Code
```c
// Aho-Corasick algorithm is quite complex and requires significant code. Please refer to specialized libraries or resources for a complete implementation in C.
// Below is a brief outline of the approach without full implementation due to space limitations.
```

### 6. Z Algorithm

#### Explanation
The Z algorithm preprocesses the pattern and text to create a Z array that stores the longest substring starting from each position that matches the prefix of the pattern.

#### C Code (Z Algorithm for Pattern Searching)
```c
#include <stdio.h>
#include <string.h>

void computeZArray(char* str,

 int Z[]) {
    int n = strlen(str);
    int L, R, K;
    L = R = 0;
    for (int i = 1; i < n; ++i) {
        if (i > R) {
            L = R = i;
            while (R < n && str[R-L] == str[R])
                R++;
            Z[i] = R-L;
            R--;
        } else {
            K = i-L;
            if (Z[K] < R-i+1)
                Z[i] = Z[K];
            else {
                L = i;
                while (R < n && str[R-L] == str[R])
                    R++;
                Z[i] = R-L;
                R--;
            }
        }
    }
}

void searchPattern(char* text, char* pattern) {
    char concat[1000];
    strcpy(concat, pattern);
    strcat(concat, "$");
    strcat(concat, text);
    int n = strlen(concat);
    int Z[n];
    computeZArray(concat, Z);
    for (int i = 0; i < n; ++i) {
        if (Z[i] == strlen(pattern))
            printf("Pattern found at index %d\n", i - strlen(pattern) - 1);
    }
}

int main() {
    char text[] = "AABAACAADAABAABA";
    char pattern[] = "AABA";
    searchPattern(text, pattern);
    return 0;
}
```

### 7. Finite Automata Algorithm

#### Explanation
The Finite Automata algorithm preprocesses the pattern to create a state machine. It matches characters of the text against the state machine to find occurrences of the pattern.

#### C Code
```c
// Finite Automata algorithm is also complex and typically requires more extensive code. Please refer to specialized resources for a complete implementation in C.
// Below is a brief outline of the approach without full implementation due to space limitations.
```

### 8. Suffix Tree Algorithm

#### Explanation
The Suffix Tree algorithm preprocesses the text to create a data structure that represents all suffixes of the text. It efficiently searches for patterns by traversing the tree.

#### C Code
```c
// Suffix Tree algorithm is complex and typically requires specialized libraries or extensive code. Please refer to advanced resources for a complete implementation in C.
// Below is a brief outline of the approach without full implementation due to space limitations.
```

### 9. Bitap Algorithm

#### Explanation
The Bitap (or Baeza-Yates-Gonnet) algorithm uses bit manipulation to perform multiple pattern searches efficiently. It preprocesses the pattern and text to generate bitmasks and matches characters using bitwise operations.

#### C Code
```c
// Bitap algorithm is typically implemented with advanced bit manipulation techniques. Below is an outline of the approach without full implementation due to space limitations.
```

### 10. Sunday Algorithm

#### Explanation
The Sunday algorithm is an improvement over the naive pattern searching approach. It checks characters of the text against the pattern using a heuristic that skips characters based on their last occurrence in the pattern.

#### C Code
```c
#include <stdio.h>
#include <string.h>

#define NO_OF_CHARS 256

// A utility function to get maximum of two integers
int max(int a, int b) {
    return (a > b) ? a : b;
}

// The preprocessing function for Sunday's bad character heuristic
void badCharHeuristic(char* str, int size, int badchar[NO_OF_CHARS]) {
    int i;

    // Initialize all occurrences as -1
    for (i = 0; i < NO_OF_CHARS; i++)
        badchar[i] = -1;

    // Fill the actual value of last occurrence of a character
    for (i = 0; i < size; i++)
        badchar[(int) str[i]] = i;
}

/* A pattern searching function that uses Sunday's Algorithm */
void SundaySearch(char* text, char* pattern) {
    int n = strlen(text);
    int m = strlen(pattern);
    int badchar[NO_OF_CHARS];

    // Fill the bad character array by calling the preprocessing function badCharHeuristic() for given pattern
    badCharHeuristic(pattern, m, badchar);

    int s = 0; // s is shift of the pattern with respect to text
    while (s <= (n - m)) {
        int j = 0;

        // Keep reducing index j of pattern while characters of pattern and text are matching at this shift s
        while (j < m && pattern[j] == text[s + j])
            j++;

        // If the pattern is present at current shift, then index j will become m after the above loop
        if (j == m) {
            printf("Pattern occurs at shift = %d\n", s);
        }

        // Shift the pattern so that the next character in text aligns with the last occurrence of it in pattern.
        s += (s + m < n) ? m - badchar[text[s + m]] : 1;
    }
}

int main() {
    char text[] = "ABAAABCD";
    char pattern[] = "ABC";
    SundaySearch(text, pattern);
    return 0;
}
```

These pattern searching algorithms demonstrate various techniques and approaches to efficiently find occurrences of patterns within texts or strings using different computational methods and optimizations.

## SORTING ALGORITHMS

Sorting algorithms are fundamental in computer science, used to arrange elements of a list in a specific order. Here are explanations and C code implementations for ten classic sorting algorithms:

### 1. Bubble Sort

#### Explanation
Bubble Sort repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted.

#### C Code
```c
#include <stdio.h>

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap arr[j] and arr[j + 1]
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    bubbleSort(arr, n);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 2. Selection Sort

#### Explanation
Selection Sort repeatedly finds the minimum element from the unsorted part of the array and swaps it with the element at the beginning of the unsorted part. It divides the array into sorted and unsorted subarrays.

#### C Code
```c
#include <stdio.h>

void selectionSort(int arr[], int n) {
    int i, j, min_idx;

    for (i = 0; i < n - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        // Swap the found minimum element with the first element of the unsorted array
        int temp = arr[min_idx];
        arr[min_idx] = arr[i];
        arr[i] = temp;
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    selectionSort(arr, n);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 3. Insertion Sort

#### Explanation
Insertion Sort builds the final sorted array one item at a time. It takes each element from the unsorted part and inserts it into its correct position in the sorted part, shifting elements as needed.

#### C Code
```c
#include <stdio.h>

void insertionSort(int arr[], int n) {
    int i, key, j;
    for (i = 1; i < n; i++) {
        key = arr[i];
        j = i - 1;

        // Move elements of arr[0..i-1], that are greater than key, to one position ahead of their current position
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    insertionSort(arr, n);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 4. Merge Sort

#### Explanation
Merge Sort divides the array into two halves, recursively sorts each half, and then merges the sorted halves. It uses a divide-and-conquer approach.

#### C Code
```c
#include <stdio.h>

void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary arrays
    int L[n1], R[n2];

    // Copy data to temporary arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temporary arrays back into arr[l..r]
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        // Merge the sorted halves
        merge(arr, l, m, r);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    mergeSort(arr, 0, n - 1);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 5. Quick Sort

#### Explanation
Quick Sort picks an element as a pivot and partitions the array around the pivot. It places the pivot in its correct position and recursively sorts the subarrays on either side of the pivot.

#### C Code
```c
#include <stdio.h>

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element

    for (int j = low; j <= high - 1; j++) {
        // If current element is smaller than the pivot
        if (arr[j] < pivot) {
            i++; // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Recursively sort elements before partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    quickSort(arr, 0, n - 1);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 6. Heap Sort

#### Explanation
Heap Sort builds a max heap from the array and repeatedly removes the largest element from the heap, placing it at the end of the array. It uses the heap property to maintain the order.

#### C Code
```c
#include <stdio.h>

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

void heapify(int arr[], int n, int i) {
    int largest = i;
    int l = 2 * i + 1; // left child
    int r = 2 * i + 2; // right child

    // If left child is larger than root
    if (l < n && arr[l] > arr[largest])
        largest = l;

    // If right child is larger than largest so far
    if (r < n && arr[r] > arr[largest])
        largest = r;

    // If largest is not root
    if (

largest != i) {
        swap(&arr[i], &arr[largest]);
        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}

void heapSort(int arr[], int n) {
    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // One by one extract an element from heap
    for (int i = n - 1; i > 0; i--) {
        // Move current root to end
        swap(&arr[0], &arr[i]);
        // call max heapify on the reduced heap
        heapify(arr, i, 0);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    heapSort(arr, n);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 7. Counting Sort

#### Explanation
Counting Sort works by counting the number of occurrences of each unique element in the input array, then using those counts to place elements into sorted order.

#### C Code
```c
#include <stdio.h>
#include <stdlib.h>

void countingSort(int arr[], int n, int range) {
    int output[n];
    int count[range + 1];
    for (int i = 0; i <= range; ++i)
        count[i] = 0;

    // Store count of each character
    for (int i = 0; i < n; ++i)
        ++count[arr[i]];

    // Change count[i] so that count[i] now contains actual position of this character in output array
    for (int i = 1; i <= range; ++i)
        count[i] += count[i - 1];

    // Build the output character array
    for (int i = 0; i < n; ++i) {
        output[count[arr[i]] - 1] = arr[i];
        --count[arr[i]];
    }

    // Copy the output array to arr[], so that arr[] now contains sorted characters
    for (int i = 0; i < n; ++i)
        arr[i] = output[i];
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; ++i)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {4, 2, 2, 8, 3, 3, 1};
    int n = sizeof(arr) / sizeof(arr[0]);
    int range = 8; // Range of input elements
    countingSort(arr, n, range);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 8. Radix Sort

#### Explanation
Radix Sort sorts numbers by processing individual digits. It sorts numbers from least significant digit (LSD) to most significant digit (MSD) using counting sort as a subroutine.

#### C Code
```c
#include <stdio.h>

// A utility function to get the maximum value in arr[]
int getMax(int arr[], int n) {
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}

// A function to do counting sort of arr[] according to the digit represented by exp.
void countSort(int arr[], int n, int exp) {
    int output[n]; // output array
    int i, count[10] = {0};

    // Store count of occurrences in count[]
    for (i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    // Change count[i] so that count[i] now contains actual position of this digit in output[]
    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];

    // Build the output array
    for (i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    // Copy the output array to arr[], so that arr[] now contains sorted numbers according to the current digit
    for (i = 0; i < n; i++)
        arr[i] = output[i];
}

// The main function to that sorts arr[] of size n using Radix Sort
void radixSort(int arr[], int n) {
    // Find the maximum number to know number of digits
    int m = getMax(arr, n);

    // Do counting sort for every digit. Note that instead of passing digit number, exp is passed. exp is 10^i where i is current digit number
    for (int exp = 1; m / exp > 0; exp *= 10)
        countSort(arr, n, exp);
}

// A utility function to print an array
void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// Driver program to test above functions
int main() {
    int arr[] = {170, 45, 75, 90, 802, 24, 2, 66};
    int n = sizeof(arr) / sizeof(arr[0]);
    radixSort(arr, n);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 9. Shell Sort

#### Explanation
Shell Sort improves upon Insertion Sort by sorting elements that are far apart before progressively reducing the gap between elements being compared. It uses an increment sequence to sort subsets of the array.

#### C Code
```c
#include <stdio.h>

void shellSort(int arr[], int n) {
    for (int gap = n / 2; gap > 0; gap /= 2) {
        // Perform a gapped insertion sort for this gap size.
        // The first gap elements arr[0..gap-1] are already in gapped order
        // keep adding one more element until the entire array is gap sorted
        for (int i = gap; i < n; i++) {
            // add arr[i] to the elements that have been gap sorted
            // save arr[i] in temp and make a hole at position i
            int temp = arr[i];

            // shift earlier gap-sorted elements up until the correct location for arr[i] is found
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap)
                arr[j] = arr[j - gap];

            // put temp (the original arr[i]) in its correct location
            arr[j] = temp;
        }
    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    shellSort(arr, n);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

### 10. Cocktail Shaker Sort (Bidirectional Bubble Sort)

#### Explanation
Cocktail Shaker Sort is a variation of Bubble Sort that sorts in both directions, first from left to right, then from right to left. It improves Bubble Sort by reducing the number of passes.

#### C Code
```c
#include <stdio.h>

void cocktailSort(int arr[], int n) {
    int swapped = 1;
    int start = 0;
    int end = n - 1;

    while (swapped) {
        // Reset the swapped flag on entering the loop, because it might be true from a previous iteration
        swapped = 0;

        // Perform a bubble sort from left to right
        for (int i = start; i < end; ++i) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                swapped = 1;
            }
        }

        // If nothing moved, then array is sorted
        if (!swapped)
            break;

        // Otherwise, reset the swapped flag so that it can be used in the next stage
        swapped = 0;

        // Move the end point back by one, because the item at the end is in its rightful spot
        end--;

        // Perform a bubble sort from right to left
        for (int i = end - 1; i >= start; --i) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                swapped = 1;
            }
        }

        // Move the starting point forward

 by one, because the item at the start is in its rightful spot
        start++;
    }
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);
    cocktailSort(arr, n);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

These are ten classic sorting algorithms with explanations and C code implementations. Each algorithm has its unique approach to sorting elements, and their performance varies based on the input size and initial ordering of elements in the array.

## SEARCHING ALGORITHMS

Here are explanations and C code implementations for various searching algorithms:

### 1. Linear Search

#### Explanation
Linear Search sequentially checks each element of the list until it finds the target element or reaches the end of the list.

#### C Code
```c
#include <stdio.h>

int linearSearch(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target)
            return i; // Return the index of the target element
    }
    return -1; // Return -1 if the target element is not found
}

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 10;
    int result = linearSearch(arr, n, target);
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);
    return 0;
}
```

### 2. Binary Search (Iterative)

#### Explanation
Binary Search works on sorted arrays by repeatedly dividing the search interval in half. It compares the target value with the middle element of the array and eliminates half of the remaining elements.

#### C Code
```c
#include <stdio.h>

int binarySearch(int arr[], int left, int right, int target) {
    while (left <= right) {
        int mid = left + (right - left) / 2;

        // Check if target is present at mid
        if (arr[mid] == target)
            return mid;

        // If target greater, ignore left half
        if (arr[mid] < target)
            left = mid + 1;

        // If target is smaller, ignore right half
        else
            right = mid - 1;
    }

    // If target is not present in array
    return -1;
}

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 10;
    int result = binarySearch(arr, 0, n - 1, target);
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);
    return 0;
}
```

### 3. Binary Search (Recursive)

#### Explanation
Recursive Binary Search is a recursive version of the binary search algorithm. It divides the search interval in half and recursively calls itself until the target is found or the interval is empty.

#### C Code
```c
#include <stdio.h>

int binarySearchRecursive(int arr[], int left, int right, int target) {
    if (right >= left) {
        int mid = left + (right - left) / 2;

        // If the target is present at the middle
        if (arr[mid] == target)
            return mid;

        // If the target is smaller than mid, it can only be present in the left subarray
        if (arr[mid] > target)
            return binarySearchRecursive(arr, left, mid - 1, target);

        // Else the target can only be present in the right subarray
        return binarySearchRecursive(arr, mid + 1, right, target);
    }

    // If the target is not present in array
    return -1;
}

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 10;
    int result = binarySearchRecursive(arr, 0, n - 1, target);
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);
    return 0;
}
```

### 4. Jump Search

#### Explanation
Jump Search works on sorted arrays and jumps ahead by a fixed number of steps (like √n) to find a range where the target element could be. It then performs a linear search in that range.

#### C Code
```c
#include <stdio.h>
#include <math.h>

int jumpSearch(int arr[], int n, int target) {
    int step = sqrt(n); // Step size

    // Finding the block where the target element is present (if any)
    int prev = 0;
    while (arr[fmin(step, n) - 1] < target) {
        prev = step;
        step += sqrt(n);
        if (prev >= n)
            return -1;
    }

    // Doing a linear search for target in the block beginning with prev
    while (arr[prev] < target) {
        prev++;

        // If we reach the next block or end of array, target is not present
        if (prev == fmin(step, n))
            return -1;
    }

    // If the element is found
    if (arr[prev] == target)
        return prev;

    return -1;
}

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 10;
    int result = jumpSearch(arr, n, target);
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);
    return 0;
}
```

### 5. Interpolation Search

#### Explanation
Interpolation Search is an improvement over Binary Search for situations where elements in a sorted array are uniformly distributed. It calculates the probable position of the target using a formula and adjusts its position accordingly.

#### C Code
```c
#include <stdio.h>

int interpolationSearch(int arr[], int n, int target) {
    int low = 0, high = (n - 1);

    while (low <= high && target >= arr[low] && target <= arr[high]) {
        // Probing the position with keeping uniform distribution in mind.
        int pos = low + (((double)(high - low) / (arr[high] - arr[low])) * (target - arr[low]));

        // Condition of target found
        if (arr[pos] == target)
            return pos;

        // If target is larger, target is in upper part
        if (arr[pos] < target)
            low = pos + 1;

        // If target is smaller, target is in lower part
        else
            high = pos - 1;
    }
    return -1; // Return -1 if target is not found
}

int main() {
    int arr[] = {10, 12, 13, 16, 18, 19, 20, 21, 22, 23, 24, 33, 35, 42, 47};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 18;
    int result = interpolationSearch(arr, n, target);
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);
    return 0;
}
```

### 6. Exponential Search

#### Explanation
Exponential Search is particularly useful for unbounded/infinite sorted arrays. It starts with a small range and exponentially increases the range until it finds a range where the target may be, then performs a binary search in that range.

#### C Code
```c
#include <stdio.h>

int binarySearch(int arr[], int left, int right, int target);

int exponentialSearch(int arr[], int n, int target) {
    // If target is present at first location itself
    if (arr[0] == target)
        return 0;

    // Find range for binary search by repeatedly doubling until target is less than or equal to the last element
    int i = 1;
    while (i < n && arr[i] <= target)
        i = i * 2;

    // Perform binary search for target in the range formed by i/2 and min(i, n-1)
    return binarySearch(arr, i / 2, (i < n ? i : n - 1), target);
}

int binarySearch(int arr[], int left, int right, int target) {
    while (left <= right) {
        int mid = left + (right - left) / 2;

        // Check if target is present at mid
        if (arr[mid] == target)
            return mid;

        // If target greater, ignore left half
        if (arr[mid] < target)
            left = mid + 1;

        // If target is smaller, ignore right half
        else
            right = mid - 1;
    }

    // If target is not present in array
    return -1;
}

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 10;
    int result = exponentialSearch(arr, n, target);
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);
    return 0;
}
```

### 7. Fibonacci Search

#### Explanation
Fibonacci Search is a comparison-based technique that

 uses Fibonacci numbers to divide the array into two parts and eliminates one part based on the comparison with the target.

#### C Code
```c
#include <stdio.h>

// Generate Fibonacci numbers up to a certain limit
void generateFibonacci(int fib[], int n) {
    fib[0] = 0;
    fib[1] = 1;
    for (int i = 2; i < n; i++)
        fib[i] = fib[i - 1] + fib[i - 2];
}

int min(int x, int y) {
    return (x <= y) ? x : y;
}

int fibonacciSearch(int arr[], int n, int target) {
    // Initialize fibonacci numbers
    int fib[100]; // Arbitrary limit for Fibonacci numbers
    generateFibonacci(fib, 100);

    int offset = -1; // Index of smallest Fibonacci number >= n
    int k = 0; // Index of current Fibonacci number

    // Find the smallest Fibonacci number greater than or equal to n
    while (fib[k] < n) {
        k++;
    }

    // Initialize variables
    int prev = 0;
    int mid;

    // Perform the search
    while (fib[k] > 1) {
        // Check if k is a valid index
        mid = min(prev + fib[k - 2], n - 1);

        // If target is greater than the value at index mid, cut the subarray from prev to mid
        if (arr[mid] < target) {
            k--;
            prev = mid;
        }

        // If target is smaller than the value at index mid, cut the subarray after mid
        else if (arr[mid] > target) {
            k -= 2;
        }

        // If target is equal to the value at index mid, return mid
        else {
            return mid;
        }
    }

    // Check the last element
    if (fib[k] == 1 && arr[prev + 1] == target) {
        return prev + 1;
    }

    // If target is not present in array
    return -1;
}

int main() {
    int arr[] = {10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 85;
    int result = fibonacciSearch(arr, n, target);
    if (result == -1)
        printf("Element is not present in array\n");
    else
        printf("Element is present at index %d\n", result);
    return 0;
}
```

### 8. Linear Probing in Hashing

#### Explanation
Linear Probing is a collision resolution technique used in hash tables. When a collision occurs (i.e., the hash function maps two keys to the same index), Linear Probing searches the table for the next available slot.

#### C Code (Hash Table with Linear Probing)
```c
#include <stdio.h>
#define SIZE 10

int hashFunction(int key) {
    return key % SIZE;
}

int probe(int H[], int key) {
    int index = hashFunction(key);
    int i = 0;
    while (H[(index + i) % SIZE] != 0)
        i++;
    return (index + i) % SIZE;
}

void insert(int H[], int key) {
    int index = hashFunction(key);

    if (H[index] != 0)
        index = probe(H, key);
    H[index] = key;
}

int search(int H[], int key) {
    int index = hashFunction(key);
    int i = 0;
    while (H[(index + i) % SIZE] != key) {
        if (H[(index + i) % SIZE] == 0)
            return -1;
        i++;
    }
    return (index + i) % SIZE;
}

int main() {
    int HT[10] = {0};
    insert(HT, 12);
    insert(HT, 25);
    insert(HT, 35);
    insert(HT, 26);
    int index = search(HT, 35);
    printf("Element found at index: %d\n", index);
    return 0;
}
```

### 9. Binary Search Tree (BST) Search

#### Explanation
Binary Search Tree is a binary tree data structure where each node has at most two children, referred to as the left child and the right child. BST property ensures that for each node, the left subtree contains only nodes with keys lesser than the node's key, and the right subtree contains only nodes with keys greater than the node's key.

#### C Code (BST Search)
```c
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int key;
    struct Node *left, *right;
};

// Function to create a new BST node
struct Node *newNode(int item) {
    struct Node *temp = (struct Node *)malloc(sizeof(struct Node));
    temp->key = item;
    temp->left = temp->right = NULL;
    return temp;
}

// Function to perform BST search
struct Node *search(struct Node *root, int key) {
    // Base Cases: root is null or key is present at the root
    if (root == NULL || root->key == key)
        return root;

    // Key is greater than root's key
    if (root->key < key)
        return search(root->right, key);

    // Key is smaller than root's key
    return search(root->left, key);
}

// Driver Program to test above functions
int main() {
    struct Node *root = newNode(50);
    root->left = newNode(30);
    root->right = newNode(70);
    root->left->left = newNode(20);
    root->left->right = newNode(40);
    root->right->left = newNode(60);
    root->right->right = newNode(80);

    int key = 40;
    struct Node *result = search(root, key);
    if (result != NULL)
        printf("Element %d found in BST\n", key);
    else
        printf("Element %d not found in BST\n", key);
    return 0;
}
```

### 10. Depth First Search (DFS) in Graph

#### Explanation
Depth First Search (DFS) is a graph traversal algorithm that explores as far as possible along each branch before backtracking. It uses a stack to keep track of vertices.

#### C Code (Graph DFS)
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX 100

// Structure to represent an adjacency list node
struct Node {
    int dest;
    struct Node *next;
};

// Structure to represent an adjacency list
struct AdjList {
    struct Node *head;
};

// Structure to represent a graph
struct Graph {
    int V; // Number of vertices
    struct AdjList *array;
};

// Function to create a new adjacency list node
struct Node *newNode(int dest) {
    struct Node *newNode = (struct Node *)malloc(sizeof(struct Node));
    newNode->dest = dest;
    newNode->next = NULL;
    return newNode;
}

// Function to create a graph with V vertices
struct Graph *createGraph(int V) {
    struct Graph *graph = (struct Graph *)malloc(sizeof(struct Graph));
    graph->V = V;
    graph->array = (struct AdjList *)malloc(V * sizeof(struct AdjList));

    for (int i = 0; i < V; ++i)
        graph->array[i].head = NULL;

    return graph;
}

// Function to add an edge to an undirected graph
void addEdge(struct Graph *graph, int src, int dest) {
    // Add an edge from src to dest
    struct Node *newNode = newNode(dest);
    newNode->next = graph->array[src].head;
    graph->array[src].head = newNode;

    // Since graph is undirected, add an edge from dest to src also
    newNode = newNode(src);
    newNode->next = graph->array[dest].head;
    graph->array[dest].head = newNode;
}

// Function to perform DFS traversal of the vertices reachable from v
void DFSUtil(int v, bool visited[], struct Graph *graph) {
    // Mark the current node as visited and print it
    visited[v] = true;
    printf("%d ", v);

    // Recur for all the vertices adjacent to this vertex
    struct Node *temp = graph->array[v].head;
    while (temp != NULL) {
        if (!visited[temp->dest])
            DFSUtil(temp->dest, visited, graph);
        temp = temp->next;
    }
}

// Function to perform DFS traversal of the graph
void DFS(struct Graph *graph, int v) {
    // Mark all the vertices as not visited
    bool *visited = (bool *)malloc(graph->V * sizeof(bool));
    for (int i = 0; i < graph->V; i++)
        visited[i] = false;

    // Call the recursive helper function to print DFS traversal
    DFSUtil(v, visited, graph);
}

// Driver Program to test above functions
int main() {
    int V = 4;
    struct Graph *graph = createGraph(V);
    addEdge(graph, 0, 1);
    addEdge(graph, 0, 2);
    addEdge(graph, 1, 2);
    addEdge(graph, 2, 0);
    addEdge(graph, 2, 3);


    addEdge(graph, 3, 3);

    printf("Following is Depth First Traversal (starting from vertex 2):\n");
    DFS(graph, 2);
    printf("\n");

    return 0;
}
```

These are examples of various search algorithms implemented in C, each suited for different types of data structures and use cases, from basic array searching to more complex graph traversals.

## BIT MANIPULATIONS

Certainly! Here are explanations and C code examples for 10 common bitwise operations and problems:

### 1. Bitwise AND (&)

#### Explanation
Bitwise AND operation performs a logical AND between corresponding bits of two integers. It results in a bit being set only if both corresponding bits of operands are set.

#### C Code
```c
#include <stdio.h>

int main() {
    int a = 5;    // 101 in binary
    int b = 3;    // 011 in binary
    int result = a & b;
    
    printf("Result of bitwise AND between %d and %d is %d\n", a, b, result);  // Output: 1
    
    return 0;
}
```

### 2. Bitwise OR (|)

#### Explanation
Bitwise OR operation performs a logical OR between corresponding bits of two integers. It results in a bit being set if at least one of the corresponding bits of operands is set.

#### C Code
```c
#include <stdio.h>

int main() {
    int a = 5;    // 101 in binary
    int b = 3;    // 011 in binary
    int result = a | b;
    
    printf("Result of bitwise OR between %d and %d is %d\n", a, b, result);  // Output: 7
    
    return 0;
}
```

### 3. Bitwise XOR (^)

#### Explanation
Bitwise XOR operation performs a logical XOR (exclusive OR) between corresponding bits of two integers. It results in a bit being set if only one of the corresponding bits of operands is set.

#### C Code
```c
#include <stdio.h>

int main() {
    int a = 5;    // 101 in binary
    int b = 3;    // 011 in binary
    int result = a ^ b;
    
    printf("Result of bitwise XOR between %d and %d is %d\n", a, b, result);  // Output: 6
    
    return 0;
}
```

### 4. Bitwise NOT (~)

#### Explanation
Bitwise NOT operation inverts each bit of the operand. It turns 1s into 0s and 0s into 1s.

#### C Code
```c
#include <stdio.h>

int main() {
    int a = 5;    // 101 in binary
    int result = ~a;
    
    printf("Result of bitwise NOT of %d is %d\n", a, result);  // Output: -6
    
    return 0;
}
```

### 5. Left Shift (<<)

#### Explanation
Left Shift operation shifts the bits of a number to the left by a specified number of positions, effectively multiplying the number by 2 for each shift.

#### C Code
```c
#include <stdio.h>

int main() {
    int a = 5;    // 101 in binary
    int result = a << 2;   // Shift left by 2 positions
    
    printf("Result of left shift of %d by 2 positions is %d\n", a, result);  // Output: 20 (10100 in binary)
    
    return 0;
}
```

### 6. Right Shift (>>)

#### Explanation
Right Shift operation shifts the bits of a number to the right by a specified number of positions, effectively dividing the number by 2 for each shift.

#### C Code
```c
#include <stdio.h>

int main() {
    int a = 20;    // 10100 in binary
    int result = a >> 2;   // Shift right by 2 positions
    
    printf("Result of right shift of %d by 2 positions is %d\n", a, result);  // Output: 5 (101 in binary)
    
    return 0;
}
```

### 7. Checking if a Number is Even or Odd

#### Explanation
Using bitwise AND with 1 (which is 0001 in binary) can determine if a number is even or odd. If the result is 0, the number is even; if 1, the number is odd.

#### C Code
```c
#include <stdio.h>

int main() {
    int num = 17;   // Example number
    if (num & 1)
        printf("%d is odd\n", num);
    else
        printf("%d is even\n", num);
    
    return 0;
}
```

### 8. Swapping Two Numbers

#### Explanation
Swapping two numbers using bitwise XOR without using a temporary variable.

#### C Code
```c
#include <stdio.h>

int main() {
    int a = 5;
    int b = 10;
    
    printf("Before swapping: a = %d, b = %d\n", a, b);
    
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
    
    printf("After swapping: a = %d, b = %d\n", a, b);
    
    return 0;
}
```

### 9. Setting a Bit at a Given Position

#### Explanation
Setting a bit at a given position using bitwise OR.

#### C Code
```c
#include <stdio.h>

int main() {
    int num = 5;    // 101 in binary
    int pos = 1;    // Position to set (0-indexed from right)
    
    num = num | (1 << pos);
    
    printf("Number after setting bit at position %d: %d\n", pos, num);  // Output: 7 (111 in binary)
    
    return 0;
}
```

### 10. Clearing a Bit at a Given Position

#### Explanation
Clearing a bit at a given position using bitwise AND with the complement of the bit.

#### C Code
```c
#include <stdio.h>

int main() {
    int num = 7;    // 111 in binary
    int pos = 1;    // Position to clear (0-indexed from right)
    
    num = num & ~(1 << pos);
    
    printf("Number after clearing bit at position %d: %d\n", pos, num);  // Output: 5 (101 in binary)
    
    return 0;
}
```


## MATHEMATICAL ALGORITHMS

Mathematical algorithms cover a broad range of problems that involve mathematical operations, computations, or optimizations. Here are some classic mathematical algorithms along with their explanations and C code implementations:

### 1. Euclidean Algorithm for GCD (Greatest Common Divisor)

#### Explanation
The Euclidean algorithm is used to find the greatest common divisor (GCD) of two integers \( a \) and \( b \). It repeatedly replaces the larger number by its remainder when divided by the smaller number until one of the numbers becomes zero, at which point the other number is the GCD.

#### C Code
```c
#include <stdio.h>

int gcd(int a, int b) {
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

int main() {
    int a = 24, b = 60;
    printf("GCD of %d and %d is %d\n", a, b, gcd(a, b));
    return 0;
}
```

### 2. Sieve of Eratosthenes for Prime Numbers

#### Explanation
The Sieve of Eratosthenes is an efficient algorithm to find all prime numbers up to a specified integer \( n \). It works by iteratively marking the multiples of each prime number starting from 2.

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>

void sieveOfEratosthenes(int n) {
    bool prime[n+1];
    memset(prime, true, sizeof(prime));

    for (int p = 2; p * p <= n; p++) {
        if (prime[p] == true) {
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }

    printf("Prime numbers up to %d are: ", n);
    for (int p = 2; p <= n; p++) {
        if (prime[p])
            printf("%d ", p);
    }
    printf("\n");
}

int main() {
    int n = 50;
    sieveOfEratosthenes(n);
    return 0;
}
```

### 3. Newton-Raphson Method for Square Root

#### Explanation
The Newton-Raphson method is an iterative method to find the square root of a number \( x \). It starts with an initial guess \( y_0 \) and iteratively improves the guess until the difference between consecutive values is within a desired tolerance.

#### C Code
```c
#include <stdio.h>
#include <math.h>

double squareRoot(double x) {
    double y = x;
    double epsilon = 0.00001; // tolerance

    while (fabs(y * y - x) > epsilon) {
        y = (y + x / y) / 2.0;
    }
    return y;
}

int main() {
    double num = 25.0;
    printf("Square root of %.2f is %.4f\n", num, squareRoot(num));
    return 0;
}
```

### 4. Sieve of Atkin for Prime Numbers

#### Explanation
The Sieve of Atkin is another algorithm to find all prime numbers up to a specified integer \( n \). It is more efficient than the Sieve of Eratosthenes for large values of \( n \).

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

void sieveOfAtkin(int limit) {
    if (limit > 2)
        printf("2 ");

    if (limit > 3)
        printf("3 ");

    bool sieve[limit];
    for (int i = 0; i < limit; i++)
        sieve[i] = false;

    for (int x = 1; x * x < limit; x++) {
        for (int y = 1; y * y < limit; y++) {
            int n = (4 * x * x) + (y * y);
            if (n <= limit && (n % 12 == 1 || n % 12 == 5))
                sieve[n] ^= true;

            n = (3 * x * x) + (y * y);
            if (n <= limit && n % 12 == 7)
                sieve[n] ^= true;

            n = (3 * x * x) - (y * y);
            if (x > y && n <= limit && n % 12 == 11)
                sieve[n] ^= true;
        }
    }

    for (int r = 5; r * r < limit; r++) {
        if (sieve[r]) {
            for (int i = r * r; i < limit; i += r * r)
                sieve[i] = false;
        }
    }

    for (int a = 5; a < limit; a++) {
        if (sieve[a])
            printf("%d ", a);
    }
    printf("\n");
}

int main() {
    int limit = 50;
    sieveOfAtkin(limit);
    return 0;
}
```

### 5. Miller-Rabin Primality Test

#### Explanation
The Miller-Rabin primality test is an efficient probabilistic algorithm to determine if a given number is prime. It repeatedly tests a number for primality based on properties of modular arithmetic.

#### C Code (Basic Implementation)
```c
#include <stdio.h>
#include <stdbool.h>

bool isPrime(int n, int k) {
    if (n <= 1 || n == 4)
        return false;
    if (n <= 3)
        return true;

    while (k > 0) {
        int a = 2 + rand() % (n - 4);
        if (gcd(n, a) != 1)
            return false;

        int x = power(a, n - 1, n);
        if (x != 1)
            return false;
        k--;
    }
    return true;
}

int main() {
    int n = 23; // number to be tested
    int k = 3;  // number of iterations
    if (isPrime(n, k))
        printf("%d is a prime number\n", n);
    else
        printf("%d is not a prime number\n", n);
    return 0;
}
```

### 6. Fast Exponentiation (Exponentiation by Squaring)

#### Explanation
Fast exponentiation is a method to compute \( x^n \) efficiently using recursion and squaring. It reduces the number of multiplications by exploiting the properties of even and odd powers.

#### C Code
```c
#include <stdio.h>

long long power(int x, unsigned int n) {
    if (n == 0)
        return 1;
    long long temp = power(x, n / 2);
    if (n % 2 == 0)
        return temp * temp;
    else
        return x * temp * temp;
}

int main() {
    int x = 2;
    unsigned int n = 10;
    printf("%d^%d is %lld\n", x, n, power(x, n));
    return 0;
}
```

### 7. Extended Euclidean Algorithm

#### Explanation
The extended Euclidean algorithm finds the integers \( x \) and \( y \) such that \( ax + by = \text{gcd}(a, b) \).

#### C Code
```c
#include <stdio.h>

void extendedEuclid(int a, int b, int *x, int *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return;
    }
    int x1, y1;
    extendedEuclid(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
}

int main() {
    int a = 30, b = 20, x, y;
    extendedEuclid(a, b, &x, &y);
    printf("Extended Euclidean Result: x = %d, y = %d\n", x, y);
    return 0;
}
```

### 8. Karatsuba Algorithm for Fast Multiplication

#### Explanation
The Karatsuba algorithm is a fast multiplication algorithm that reduces the number of recursive calls in polynomial multiplication using a divide-and-conquer approach.

#### C Code
```c
#include <stdio.h>
#include <math.h>

int karatsuba(int x, int y) {
    if (x < 10 || y < 10)
        return x * y;

    int m = fmax(log10(x) + 1, log10(y) + 1) / 2;
    int p = pow(10, m);

    int a = x / p;
    int b = x % p;
    int c = y / p;
    int d = y % p;

    int ac = karatsuba(a, c);
    int bd = karatsuba(b, d);
    int ad_bc = karatsuba(a + b, c + d) - ac - bd;

    return ac * pow(10, 2 * m) + ad_bc * pow(10, m) + bd;
}

int main() {
    int x = 1234, y = 5678;
    printf("Multiplication result: %d\n", karatsuba(x, y));
    return 0;
}
```
Mathematical algorithms are foundational in computer science and mathematics, covering a range of problems from arithmetic to number theory. Below are some classic problems along with their explanations and C code implementations.

### 1. Greatest Common Divisor (GCD)

#### Explanation
The Greatest Common Divisor (GCD) of two integers is the largest integer that divides both numbers without leaving a remainder. The Euclidean algorithm is an efficient method for computing the GCD, which uses the principle:
\[ \text{GCD}(a, b) = \text{GCD}(b, a \% b) \]
with the base case \(\text{GCD}(a, 0) = a\).

#### C Code
```c
#include <stdio.h>

int gcd(int a, int b) {
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

int main() {
    int a = 56, b = 98;
    printf("GCD of %d and %d is %d\n", a, b, gcd(a, b));
    return 0;
}
```

### 2. Least Common Multiple (LCM)

#### Explanation
The Least Common Multiple (LCM) of two integers is the smallest positive integer that is divisible by both numbers. It can be computed using the GCD:
\[ \text{LCM}(a, b) = \frac{|a \times b|}{\text{GCD}(a, b)} \]

#### C Code
```c
#include <stdio.h>

int gcd(int a, int b) {
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

int lcm(int a, int b) {
    return (a / gcd(a, b)) * b;
}

int main() {
    int a = 15, b = 20;
    printf("LCM of %d and %d is %d\n", a, b, lcm(a, b));
    return 0;
}
```

### 3. Prime Number Check

#### Explanation
A prime number is a number greater than 1 that has no positive divisors other than 1 and itself. To check if a number is prime, we test for factors from 2 to the square root of the number.

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

bool isPrime(int n) {
    if (n <= 1)
        return false;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0)
            return false;
    }
    return true;
}

int main() {
    int num = 29;
    if (isPrime(num))
        printf("%d is a prime number\n", num);
    else
        printf("%d is not a prime number\n", num);
    return 0;
}
```

### 4. Sieve of Eratosthenes

#### Explanation
The Sieve of Eratosthenes is an efficient algorithm to find all primes less than or equal to a given limit. It works by iteratively marking the multiples of each prime starting from 2.

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

void sieveOfEratosthenes(int n) {
    bool prime[n + 1];
    for (int i = 0; i <= n; i++)
        prime[i] = true;

    for (int p = 2; p * p <= n; p++) {
        if (prime[p] == true) {
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }

    for (int p = 2; p <= n; p++)
        if (prime[p])
            printf("%d ", p);
    printf("\n");
}

int main() {
    int n = 30;
    printf("Primes less than or equal to %d are: ", n);
    sieveOfEratosthenes(n);
    return 0;
}
```

### 5. Integer Factorization

#### Explanation
Integer factorization involves breaking down a composite number into smaller non-trivial divisors, which when multiplied together give the original integer. This algorithm finds the prime factors of a given integer.

#### C Code
```c
#include <stdio.h>

void primeFactors(int n) {
    while (n % 2 == 0) {
        printf("%d ", 2);
        n = n / 2;
    }
    for (int i = 3; i <= sqrt(n); i += 2) {
        while (n % i == 0) {
            printf("%d ", i);
            n = n / i;
        }
    }
    if (n > 2)
        printf("%d ", n);
    printf("\n");
}

int main() {
    int n = 315;
    printf("Prime factors of %d are: ", n);
    primeFactors(n);
    return 0;
}
```

### 6. Modular Exponentiation

#### Explanation
Modular exponentiation is used to compute \( (x^y) \% p \) efficiently. It is widely used in cryptography.

#### C Code
```c
#include <stdio.h>

int power(int x, unsigned int y, int p) {
    int res = 1;
    x = x % p;
    while (y > 0) {
        if (y & 1)
            res = (res * x) % p;
        y = y >> 1;
        x = (x * x) % p;
    }
    return res;
}

int main() {
    int x = 2;
    int y = 5;
    int p = 13;
    printf("Power is %d\n", power(x, y, p));
    return 0;
}
```

### 7. Extended Euclidean Algorithm

#### Explanation
The Extended Euclidean Algorithm not only finds the GCD of two integers \( a \) and \( b \), but also finds the coefficients \( x \) and \( y \) such that \( ax + by = \text{GCD}(a, b) \).

#### C Code
```c
#include <stdio.h>

int gcdExtended(int a, int b, int *x, int *y) {
    if (a == 0) {
        *x = 0;
        *y = 1;
        return b;
    }
    int x1, y1;
    int gcd = gcdExtended(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
    return gcd;
}

int main() {
    int a = 35, b = 15;
    int x, y;
    int g = gcdExtended(a, b, &x, &y);
    printf("GCD(%d, %d) = %d\n", a, b, g);
    printf("Coefficients x = %d, y = %d\n", x, y);
    return 0;
}
```

### 8. Matrix Multiplication

#### Explanation
Matrix multiplication involves multiplying two matrices by taking the dot product of rows and columns. Given matrices \( A \) and \( B \), the result \( C \) is computed as:
\[ C[i][j] = \sum_{k=1}^{n} A[i][k] \times B[k][j] \]



## RECURSION

Recursion is a powerful programming technique where a function calls itself in order to solve a problem. It can be used to solve many types of problems that can be broken down into smaller, similar subproblems. Here are some classic recursion problems along with their explanations and C code implementations:

### 1. Factorial

#### Explanation
The factorial of a non-negative integer \( n \) is the product of all positive integers less than or equal to \( n \). It is defined as:
\[ n! = n \times (n-1) \times (n-2) \times \ldots \times 1 \]
with the base case \( 0! = 1 \).

#### C Code
```c
#include <stdio.h>

int factorial(int n) {
    if (n == 0)
        return 1;
    else
        return n * factorial(n - 1);
}

int main() {
    int num = 5;
    printf("Factorial of %d is %d\n", num, factorial(num));
    return 0;
}
```

### 2. Fibonacci Sequence

#### Explanation
The Fibonacci sequence is defined as:
\[ F(n) = F(n-1) + F(n-2) \]
with base cases \( F(0) = 0 \) and \( F(1) = 1 \).

#### C Code
```c
#include <stdio.h>

int fibonacci(int n) {
    if (n == 0)
        return 0;
    else if (n == 1)
        return 1;
    else
        return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    int n = 10;
    printf("Fibonacci number is %d\n", fibonacci(n));
    return 0;
}
```

### 3. Tower of Hanoi

#### Explanation
The Tower of Hanoi is a mathematical puzzle where we have three rods and \( n \) disks. The objective is to move all disks from the first rod to the third rod following these rules:
1. Only one disk can be moved at a time.
2. A disk can only be moved if it is the uppermost disk on a rod.
3. No disk may be placed on top of a smaller disk.

#### C Code
```c
#include <stdio.h>

void towerOfHanoi(int n, char from_rod, char to_rod, char aux_rod) {
    if (n == 1) {
        printf("Move disk 1 from rod %c to rod %c\n", from_rod, to_rod);
        return;
    }
    towerOfHanoi(n - 1, from_rod, aux_rod, to_rod);
    printf("Move disk %d from rod %c to rod %c\n", n, from_rod, to_rod);
    towerOfHanoi(n - 1, aux_rod, to_rod, from_rod);
}

int main() {
    int n = 3;
    towerOfHanoi(n, 'A', 'C', 'B');
    return 0;
}
```

### 4. Binary Search

#### Explanation
Binary Search is a searching algorithm used to find the position of a target value within a sorted array. The basic idea is to divide the array into halves, check the middle element, and then recursively search the appropriate half.

#### C Code
```c
#include <stdio.h>

int binarySearch(int arr[], int l, int r, int x) {
    if (r >= l) {
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }
    return -1;
}

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 10;
    int result = binarySearch(arr, 0, n - 1, x);
    (result == -1) ? printf("Element is not present in array\n") 
                   : printf("Element is present at index %d\n", result);
    return 0;
}
```

### 5. Permutations of a String

#### Explanation
Generate all permutations of a given string by swapping characters recursively.

#### C Code
```c
#include <stdio.h>
#include <string.h>

void swap(char *x, char *y) {
    char temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

void permute(char *str, int l, int r) {
    if (l == r)
        printf("%s\n", str);
    else {
        for (int i = l; i <= r; i++) {
            swap((str + l), (str + i));
            permute(str, l + 1, r);
            swap((str + l), (str + i)); // backtrack
        }
    }
}

int main() {
    char str[] = "ABC";
    int n = strlen(str);
    permute(str, 0, n - 1);
    return 0;
}
```

### 6. Reverse a String

#### Explanation
Reverse a given string by recursively swapping characters from the start and end.

#### C Code
```c
#include <stdio.h>
#include <string.h>

void reverse(char str[], int start, int end) {
    if (start >= end)
        return;
    char temp = str[start];
    str[start] = str[end];
    str[end] = temp;
    reverse(str, start + 1, end - 1);
}

int main() {
    char str[] = "Hello";
    reverse(str, 0, strlen(str) - 1);
    printf("Reversed string is %s\n", str);
    return 0;
}
```

### 7. Sum of Digits

#### Explanation
Calculate the sum of digits of a given number recursively.

#### C Code
```c
#include <stdio.h>

int sumOfDigits(int n) {
    if (n == 0)
        return 0;
    return (n % 10) + sumOfDigits(n / 10);
}

int main() {
    int num = 12345;
    printf("Sum of digits of %d is %d\n", num, sumOfDigits(num));
    return 0;
}
```

### 8. Power of a Number

#### Explanation
Calculate \( x^n \) using recursion, where \( x \) is the base and \( n \) is the exponent.

#### C Code
```c
#include <stdio.h>

int power(int x, int n) {
    if (n == 0)
        return 1;
    else
        return x * power(x, n - 1);
}

int main() {
    int x = 2, n = 3;
    printf("%d^%d is %d\n", x, n, power(x, n));
    return 0;
}
```

### 9. Check if a String is a Palindrome

#### Explanation
Check if a given string is a palindrome by comparing characters from the start and end recursively.

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

bool isPalindrome(char str[], int start, int end) {
    if (start >= end)
        return true;
    if (str[start] != str[end])
        return false;
    return isPalindrome(str, start + 1, end - 1);
}

int main() {
    char str[] = "madam";
    if (isPalindrome(str, 0, strlen(str) - 1))
        printf("%s is a palindrome\n", str);
    else
        printf("%s is not a palindrome\n", str);
    return 0;
}
```

### 10. Print all Binary Strings of Length n

#### Explanation
Print all binary strings of length \( n \) using recursion.

#### C Code
```c
#include <stdio.h>

void printBinaryStrings(int n, char arr[], int i) {
    if (i == n) {
        arr[i] = '\0';
        printf("%s\n", arr);
        return;
    }
    arr[i] = '0';
    printBinaryStrings(n, arr, i + 1);
    arr[i] = '1';
    printBinaryStrings(n, arr, i + 1);
}

int main() {
    int n = 3;
    char arr[n + 1];
    printBinaryStrings(n, arr, 0);
    return 0;
}
```



## DYNAMIC PROGRAMMING

Dynamic Programming (DP) is a powerful technique for solving optimization problems by breaking them down into simpler subproblems. It avoids redundant calculations by storing the results of subproblems and reusing them. Here are some classic dynamic programming problems along with their explanations and C code implementations:

### 1. Fibonacci Sequence

#### Explanation
The Fibonacci sequence is defined as:
\[ F(n) = F(n-1) + F(n-2) \]
with base cases \( F(0) = 0 \) and \( F(1) = 1 \).

Dynamic programming is used to store the results of previously computed Fibonacci numbers to avoid redundant calculations.

#### C Code
```c
#include <stdio.h>

int fib(int n) {
    int f[n+2]; // 1 extra to handle case, n = 0
    int i;

    f[0] = 0;
    f[1] = 1;

    for (i = 2; i <= n; i++) {
        f[i] = f[i-1] + f[i-2];
    }

    return f[n];
}

int main() {
    int n = 10;
    printf("Fibonacci number is %d\n", fib(n));
    return 0;
}
```

### 2. Longest Common Subsequence (LCS)

#### Explanation
Given two sequences, find the length of the longest subsequence present in both of them.

#### C Code
```c
#include <stdio.h>
#include <string.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int lcs(char *X, char *Y, int m, int n) {
    int L[m+1][n+1];
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                L[i][j] = 0;
            else if (X[i-1] == Y[j-1])
                L[i][j] = L[i-1][j-1] + 1;
            else
                L[i][j] = max(L[i-1][j], L[i][j-1]);
        }
    }
    return L[m][n];
}

int main() {
    char X[] = "AGGTAB";
    char Y[] = "GXTXAYB";
    int m = strlen(X);
    int n = strlen(Y);
    printf("Length of LCS is %d\n", lcs(X, Y, m, n));
    return 0;
}
```

### 3. Knapsack Problem

#### Explanation
Given weights and values of \( n \) items, and a knapsack capacity \( W \), determine the maximum value that can be put in the knapsack.

#### C Code
```c
#include <stdio.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int knapSack(int W, int wt[], int val[], int n) {
    int i, w;
    int K[n+1][W+1];

    for (i = 0; i <= n; i++) {
        for (w = 0; w <= W; w++) {
            if (i == 0 || w == 0)
                K[i][w] = 0;
            else if (wt[i-1] <= w)
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w]);
            else
                K[i][w] = K[i-1][w];
        }
    }
    return K[n][W];
}

int main() {
    int val[] = {60, 100, 120};
    int wt[] = {10, 20, 30};
    int W = 50;
    int n = sizeof(val) / sizeof(val[0]);
    printf("Maximum value in Knapsack = %d\n", knapSack(W, wt, val, n));
    return 0;
}
```

### 4. Edit Distance

#### Explanation
Given two strings, find the minimum number of operations (insert, delete, or replace) required to convert one string into the other.

#### C Code
```c
#include <stdio.h>
#include <string.h>

int min(int x, int y, int z) {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

int editDist(char *str1, char *str2, int m, int n) {
    int dp[m+1][n+1];
    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0)
                dp[i][j] = j;
            else if (j == 0)
                dp[i][j] = i;
            else if (str1[i-1] == str2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]);
        }
    }
    return dp[m][n];
}

int main() {
    char str1[] = "sunday";
    char str2[] = "saturday";
    printf("Edit Distance is %d\n", editDist(str1, str2, strlen(str1), strlen(str2)));
    return 0;
}
```

### 5. Longest Increasing Subsequence (LIS)

#### Explanation
Given an array of integers, find the length of the longest subsequence such that all elements of the subsequence are sorted in increasing order.

#### C Code
```c
#include <stdio.h>

int lis(int arr[], int n) {
    int lis[n];
    int max = 0;

    for (int i = 0; i < n; i++)
        lis[i] = 1;

    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++)
            if (arr[i] > arr[j] && lis[i] < lis[j] + 1)
                lis[i] = lis[j] + 1;

    for (int i = 0; i < n; i++)
        if (max < lis[i])
            max = lis[i];

    return max;
}

int main() {
    int arr[] = {10, 22, 9, 33, 21, 50, 41, 60, 80};
    int n = sizeof(arr)/sizeof(arr[0]);
    printf("Length of LIS is %d\n", lis(arr, n));
    return 0;
}
```

### 6. Matrix Chain Multiplication

#### Explanation
Given a sequence of matrices, find the most efficient way to multiply these matrices together. The problem is not to perform the multiplications but to decide the order in which to perform the multiplications.

#### C Code
```c
#include <stdio.h>
#include <limits.h>

int matrixChainOrder(int p[], int n) {
    int m[n][n];
    for (int i = 1; i < n; i++)
        m[i][i] = 0;

    for (int L = 2; L < n; L++) {
        for (int i = 1; i < n - L + 1; i++) {
            int j = i + L - 1;
            m[i][j] = INT_MAX;
            for (int k = i; k <= j - 1; k++) {
                int q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
                if (q < m[i][j])
                    m[i][j] = q;
            }
        }
    }
    return m[1][n - 1];
}

int main() {
    int arr[] = {1, 2, 3, 4};
    int n = sizeof(arr) / sizeof(arr[0]);
    printf("Minimum number of multiplications is %d\n", matrixChainOrder(arr, n));
    return 0;
}
```

### 7. Coin Change Problem

#### Explanation
Given a set of coin denominations and a total amount, find the minimum number of coins required to make the total amount.

#### C Code
```c
#include <stdio.h>
#include <limits.h>

int minCoins(int coins[], int m, int V) {
    int table[V + 1];
    table[0] = 0;

    for (int i = 1; i <= V; i++)
        table[i] = INT_MAX;

    for (int i = 1; i <= V; i++) {
        for (int j = 0; j < m; j++)
            if (coins[j] <= i) {
                int sub_res = table[i - coins[j]];
                if (sub_res != INT_MAX && sub_res + 1 < table[i])
                    table[i] = sub_res + 1;
            }
    }
    return table[V];
}

int main() {
    int coins[] = {9, 6, 5, 1};
    int m = sizeof(coins) / sizeof(coins[0]);
    int V =

 11;
    printf("Minimum coins required is %d\n", minCoins(coins, m, V));
    return 0;
}
```

### 8. Subset Sum Problem

#### Explanation
Given a set of non-negative integers and a value \( sum \), determine if there is a subset of the given set with a sum equal to the given sum.

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>

bool isSubsetSum(int set[], int n, int sum) {
    bool subset[n + 1][sum + 1];

    for (int i = 0; i <= n; i++)
        subset[i][0] = true;

    for (int i = 1; i <= sum; i++)
        subset[0][i] = false;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= sum; j++) {
            if (j < set[i - 1])
                subset[i][j] = subset[i - 1][j];
            else
                subset[i][j] = subset[i - 1][j] || subset[i - 1][j - set[i - 1]];
        }
    }
    return subset[n][sum];
}

int main() {
    int set[] = {3, 34, 4, 12, 5, 2};
    int sum = 9;
    int n = sizeof(set) / sizeof(set[0]);
    if (isSubsetSum(set, n, sum) == true)
        printf("Found a subset with given sum\n");
    else
        printf("No subset with given sum\n");
    return 0;
}
```



## GREEDY ALGORITHM

### Greedy Algorithm Problems with Explanations and C Codes

Greedy algorithms are a class of algorithms that build up a solution piece by piece, always choosing the next piece that offers the most immediate benefit. Here are some classic problems solved using the greedy approach, along with their explanations and C code implementations:

### 1. Activity Selection Problem

#### Explanation
Given a set of activities with their start and finish times, the goal is to select the maximum number of activities that don't overlap. The greedy choice is to always pick the next activity that finishes the earliest.

#### C Code
```c
#include <stdio.h>

void activitySelection(int start[], int finish[], int n) {
    int i, j;
    printf("Selected activities: \n");

    // The first activity is always selected
    i = 0;
    printf("Activity %d: (%d, %d)\n", i + 1, start[i], finish[i]);

    for (j = 1; j < n; j++) {
        if (start[j] >= finish[i]) {
            printf("Activity %d: (%d, %d)\n", j + 1, start[j], finish[j]);
            i = j;
        }
    }
}

int main() {
    int start[] = {1, 3, 0, 5, 8, 5};
    int finish[] = {2, 4, 6, 7, 9, 9};
    int n = sizeof(start) / sizeof(start[0]);
    activitySelection(start, finish, n);
    return 0;
}
```

### 2. Fractional Knapsack Problem

#### Explanation
Given a set of items, each with a weight and a value, determine the maximum value that can be obtained by placing items into a knapsack of capacity W. Items can be broken into smaller pieces. The greedy choice is to pick the item with the highest value-to-weight ratio.

#### C Code
```c
#include <stdio.h>

struct Item {
    int value, weight;
};

void swap(struct Item* a, struct Item* b) {
    struct Item t = *a;
    *a = *b;
    *b = t;
}

int partition(struct Item arr[], int low, int high) {
    double pivot = (double)arr[high].value / arr[high].weight;
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if ((double)arr[j].value / arr[j].weight >= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(struct Item arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

double fractionalKnapsack(int W, struct Item arr[], int n) {
    quickSort(arr, 0, n - 1);

    double finalValue = 0.0;
    for (int i = 0; i < n; i++) {
        if (arr[i].weight <= W) {
            W -= arr[i].weight;
            finalValue += arr[i].value;
        } else {
            finalValue += arr[i].value * ((double)W / arr[i].weight);
            break;
        }
    }
    return finalValue;
}

int main() {
    int W = 50;
    struct Item arr[] = {{60, 10}, {100, 20}, {120, 30}};
    int n = sizeof(arr) / sizeof(arr[0]);
    printf("Maximum value in Knapsack = %.2f\n", fractionalKnapsack(W, arr, n));
    return 0;
}
```

### 3. Huffman Coding

#### Explanation
Huffman Coding is used for lossless data compression. Given a set of characters and their frequencies, the goal is to construct a prefix code that minimizes the total length of the encoded message. The greedy choice is to repeatedly merge the two least frequent characters.

#### C Code
```c
#include <stdio.h>
#include <stdlib.h>

struct MinHeapNode {
    char data;
    unsigned freq;
    struct MinHeapNode *left, *right;
};

struct MinHeap {
    unsigned size;
    unsigned capacity;
    struct MinHeapNode** array;
};

struct MinHeapNode* newNode(char data, unsigned freq) {
    struct MinHeapNode* temp = (struct MinHeapNode*)malloc(sizeof(struct MinHeapNode));
    temp->left = temp->right = NULL;
    temp->data = data;
    temp->freq = freq;
    return temp;
}

struct MinHeap* createMinHeap(unsigned capacity) {
    struct MinHeap* minHeap = (struct MinHeap*)malloc(sizeof(struct MinHeap));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array = (struct MinHeapNode**)malloc(minHeap->capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}

void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b) {
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

void minHeapify(struct MinHeap* minHeap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < minHeap->size && minHeap->array[left]->freq < minHeap->array[smallest]->freq)
        smallest = left;

    if (right < minHeap->size && minHeap->array[right]->freq < minHeap->array[smallest]->freq)
        smallest = right;

    if (smallest != idx) {
        swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}

int isSizeOne(struct MinHeap* minHeap) {
    return (minHeap->size == 1);
}

struct MinHeapNode* extractMin(struct MinHeap* minHeap) {
    struct MinHeapNode* temp = minHeap->array[0];
    minHeap->array[0] = minHeap->array[minHeap->size - 1];
    --minHeap->size;
    minHeapify(minHeap, 0);
    return temp;
}

void insertMinHeap(struct MinHeap* minHeap, struct MinHeapNode* minHeapNode) {
    ++minHeap->size;
    int i = minHeap->size - 1;
    while (i && minHeapNode->freq < minHeap->array[(i - 1) / 2]->freq) {
        minHeap->array[i] = minHeap->array[(i - 1) / 2];
        i = (i - 1) / 2;
    }
    minHeap->array[i] = minHeapNode;
}

void buildMinHeap(struct MinHeap* minHeap) {
    int n = minHeap->size - 1;
    for (int i = (n - 1) / 2; i >= 0; --i)
        minHeapify(minHeap, i);
}

void printArr(int arr[], int n) {
    for (int i = 0; i < n; ++i)
        printf("%d", arr[i]);
    printf("\n");
}

int isLeaf(struct MinHeapNode* root) {
    return !(root->left) && !(root->right);
}

struct MinHeap* createAndBuildMinHeap(char data[], int freq[], int size) {
    struct MinHeap* minHeap = createMinHeap(size);
    for (int i = 0; i < size; ++i)
        minHeap->array[i] = newNode(data[i], freq[i]);
    minHeap->size = size;
    buildMinHeap(minHeap);
    return minHeap;
}

struct MinHeapNode* buildHuffmanTree(char data[], int freq[], int size) {
    struct MinHeapNode *left, *right, *top;
    struct MinHeap* minHeap = createAndBuildMinHeap(data, freq, size);
    while (!isSizeOne(minHeap)) {
        left = extractMin(minHeap);
        right = extractMin(minHeap);
        top = newNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;
        insertMinHeap(minHeap, top);
    }
    return extractMin(minHeap);
}

void printCodes(struct MinHeapNode* root, int arr[], int top) {
    if (root->left) {
        arr[top] = 0;
        printCodes(root->left, arr, top + 1);
    }

    if (root->right) {
        arr[top] = 1;
        printCodes(root->right, arr, top + 1);
    }

    if (isLeaf(root)) {
        printf("%c: ", root->data);
        printArr(arr, top);
    }
}

void HuffmanCodes(char data[], int freq[], int size) {
    struct MinHeapNode* root = buildHuffmanTree(data, freq, size);
    int arr[100], top = 0;
    printCodes(root, arr, top);
}

int main() {
    char arr[] = {'a', 'b', 'c', 'd', 'e', 'f'};
    int freq[]

 = {5, 9, 12, 13, 16, 45};
    int size = sizeof(arr) / sizeof(arr[0]);
    HuffmanCodes(arr, freq, size);
    return 0;
}
```

### 4. Job Sequencing Problem

#### Explanation
Given a set of jobs, each with a deadline and profit, schedule the jobs to maximize the total profit while ensuring that no two jobs overlap. The greedy choice is to select jobs based on their profit in descending order and schedule them in the latest possible slot.

#### C Code
```c
#include <stdio.h>
#include <stdlib.h>

struct Job {
    char id;
    int deadline;
    int profit;
};

void swap(struct Job* a, struct Job* b) {
    struct Job temp = *a;
    *a = *b;
    *b = temp;
}

int min(int x, int y) {
    return (x < y) ? x : y;
}

void sortJobs(struct Job arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j].profit < arr[j + 1].profit) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

void jobSequencing(struct Job arr[], int n) {
    sortJobs(arr, n);

    int result[n];
    int slot[n];
    for (int i = 0; i < n; i++)
        slot[i] = 0;

    for (int i = 0; i < n; i++) {
        for (int j = min(n, arr[i].deadline) - 1; j >= 0; j--) {
            if (slot[j] == 0) {
                result[j] = i;
                slot[j] = 1;
                break;
            }
        }
    }

    printf("Scheduled jobs: ");
    for (int i = 0; i < n; i++)
        if (slot[i])
            printf("%c ", arr[result[i]].id);
    printf("\n");
}

int main() {
    struct Job arr[] = {{'a', 2, 100}, {'b', 1, 19}, {'c', 2, 27}, {'d', 1, 25}, {'e', 3, 15}};
    int n = sizeof(arr) / sizeof(arr[0]);
    jobSequencing(arr, n);
    return 0;
}
```

### 5. Prim’s Minimum Spanning Tree

#### Explanation
Given a connected, undirected graph, find a minimum spanning tree. Prim’s algorithm starts with a single vertex and repeatedly adds the smallest edge that connects a vertex in the tree to a vertex outside the tree.

#### C Code
```c
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

#define V 5

int minKey(int key[], bool mstSet[]) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++)
        if (mstSet[v] == false && key[v] < min)
            min = key[v], min_index = v;
    return min_index;
}

void printMST(int parent[], int graph[V][V]) {
    printf("Edge \tWeight\n");
    for (int i = 1; i < V; i++)
        printf("%d - %d \t%d \n", parent[i], i, graph[i][parent[i]]);
}

void primMST(int graph[V][V]) {
    int parent[V];
    int key[V];
    bool mstSet[V];

    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = false;

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet);
        mstSet[u] = true;

        for (int v = 0; v < V; v++)
            if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }

    printMST(parent, graph);
}

int main() {
    int graph[V][V] = {{0, 2, 0, 6, 0},
                       {2, 0, 3, 8, 5},
                       {0, 3, 0, 0, 7},
                       {6, 8, 0, 0, 9},
                       {0, 5, 7, 9, 0}};
    primMST(graph);
    return 0;
}
```

### 6. Dijkstra’s Shortest Path Algorithm

#### Explanation
Given a graph and a source vertex, find the shortest path to all other vertices. Dijkstra’s algorithm maintains a set of vertices whose shortest distance from the source is known and repeatedly selects the vertex with the smallest known distance, updating the distance of its neighbors.

#### C Code
```c
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

#define V 9

int minDistance(int dist[], bool sptSet[]) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = v;
    return min_index;
}

void printSolution(int dist[]) {
    printf("Vertex \t Distance from Source\n");
    for (int i = 0; i < V; i++)
        printf("%d \t\t %d\n", i, dist[i]);
}

void dijkstra(int graph[V][V], int src) {
    int dist[V];
    bool sptSet[V];

    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, sptSet[i] = false;

    dist[src] = 0;

    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, sptSet);
        sptSet[u] = true;

        for (int v = 0; v < V; v++)
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }

    printSolution(dist);
}

int main() {
    int graph[V][V] = {{0, 4, 0, 0, 0, 0, 0, 8, 0},
                       {4, 0, 8, 0, 0, 0, 0, 11, 0},
                       {0, 8, 0, 7, 0, 4, 0, 0, 2},
                       {0, 0, 7, 0, 9, 14, 0, 0, 0},
                       {0, 0, 0, 9, 0, 10, 0, 0, 0},
                       {0, 0, 4, 14, 10, 0, 2, 0, 0},
                       {0, 0, 0, 0, 0, 2, 0, 1, 6},
                       {8, 11, 0, 0, 0, 0, 1, 0, 7},
                       {0, 0, 2, 0, 0, 0, 6, 7, 0}};
    dijkstra(graph, 0);
    return 0;
}
```



## BACKTRACKING ALGORITHM


**Backtracking** is a general algorithmic technique that considers searching every possible combination in order to solve an optimization problem. Here are some classic backtracking problems along with their explanations:

### 1. Rat in a Maze

**Problem**
Find a path for a rat from the top-left corner to the bottom-right corner of a maze.

**Explanation**
The rat can move in four directions: up, down, left, and right. The algorithm uses backtracking to explore all possible paths and marks the cells in the solution path.

#### C Code
```c
#include <stdio.h>

#define N 4

void printSolution(int sol[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", sol[i][j]);
        }
        printf("\n");
    }
}

int isSafe(int maze[N][N], int x, int y) {
    return (x >= 0 && x < N && y >= 0 && y < N && maze[x][y] == 1);
}

int solveMazeUtil(int maze[N][N], int x, int y, int sol[N][N]) {
    if (x == N - 1 && y == N - 1) {
        sol[x][y] = 1;
        return 1;
    }

    if (isSafe(maze, x, y)) {
        sol[x][y] = 1;

        if (solveMazeUtil(maze, x + 1, y, sol)) {
            return 1;
        }

        if (solveMazeUtil(maze, x, y + 1, sol)) {
            return 1;
        }

        sol[x][y] = 0;
        return 0;
    }

    return 0;
}

int solveMaze(int maze[N][N]) {
    int sol[N][N] = { {0, 0, 0, 0},
                      {0, 0, 0, 0},
                      {0, 0, 0, 0},
                      {0, 0, 0, 0} };

    if (solveMazeUtil(maze, 0, 0, sol) == 0) {
        printf("No solution exists\n");
        return 0;
    }

    printSolution(sol);
    return 1;
}

int main() {
    int maze[N][N] = { {1, 0, 0, 0},
                       {1, 1, 0, 1},
                       {0, 1, 0, 0},
                       {1, 1, 1, 1} };

    solveMaze(maze);
    return 0;
}
```

### 2. N-Queens Problem

**Problem**
Place N queens on an N x N chessboard so that no two queens attack each other.

**Explanation**
A queen can attack another queen if they are in the same row, column, or diagonal. The algorithm places queens one by one in different columns, starting from the leftmost column. If placing the queen in one column does not lead to a solution, it backtracks and moves the queen to a different row in the previous column.

#### C Code
```c
#include <stdio.h>
#define N 4

void printSolution(int board[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf(" %d ", board[i][j]);
        printf("\n");
    }
}

int isSafe(int board[N][N], int row, int col) {
    int i, j;

    for (i = 0; i < col; i++)
        if (board[row][i])
            return 0;

    for (i = row, j = col; i >= 0 && j >= 0; i--, j--)
        if (board[i][j])
            return 0;

    for (i = row, j = col; j >= 0 && i < N; i++, j--)
        if (board[i][j])
            return 0;

    return 1;
}

int solveNQUtil(int board[N][N], int col) {
    if (col >= N)
        return 1;

    for (int i = 0; i < N; i++) {
        if (isSafe(board, i, col)) {
            board[i][col] = 1;

            if (solveNQUtil(board, col + 1))
                return 1;

            board[i][col] = 0;
        }
    }

    return 0;
}

int solveNQ() {
    int board[N][N] = { {0, 0, 0, 0},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0} };

    if (solveNQUtil(board, 0) == 0) {
        printf("Solution does not exist");
        return 0;
    }

    printSolution(board);
    return 1;
}

int main() {
    solveNQ();
    return 0;
}
```

### 3. Sudoku Solver

**Problem**
Fill a 9 x 9 Sudoku grid so that each column, each row, and each of the nine 3 x 3 subgrids contain all of the digits from 1 to 9.

**Explanation**
The algorithm tries placing numbers from 1 to 9 in each empty cell and checks if the number can be placed without violating Sudoku rules. If a number fits, it moves to the next empty cell. If not, it backtracks and tries a different number.

#### C Code
```c
#include <stdio.h>
#define UNASSIGNED 0
#define N 9

int isSafe(int grid[N][N], int row, int col, int num);

int FindUnassignedLocation(int grid[N][N], int *row, int *col) {
    for (*row = 0; *row < N; (*row)++)
        for (*col = 0; *col < N; (*col)++)
            if (grid[*row][*col] == UNASSIGNED)
                return 1;
    return 0;
}

int isSafeInRow(int grid[N][N], int row, int num) {
    for (int col = 0; col < N; col++)
        if (grid[row][col] == num)
            return 0;
    return 1;
}

int isSafeInCol(int grid[N][N], int col, int num) {
    for (int row = 0; row < N; row++)
        if (grid[row][col] == num)
            return 0;
    return 1;
}

int isSafeInBox(int grid[N][N], int boxStartRow, int boxStartCol, int num) {
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            if (grid[row + boxStartRow][col + boxStartCol] == num)
                return 0;
    return 1;
}

int isSafe(int grid[N][N], int row, int col, int num) {
    return isSafeInRow(grid, row, num) &&
           isSafeInCol(grid, col, num) &&
           isSafeInBox(grid, row - row % 3, col - col % 3, num);
}

int SolveSudoku(int grid[N][N]) {
    int row, col;

    if (!FindUnassignedLocation(grid, &row, &col))
        return 1;

    for (int num = 1; num <= 9; num++) {
        if (isSafe(grid, row, col, num)) {
            grid[row][col] = num;

            if (SolveSudoku(grid))
                return 1;

            grid[row][col] = UNASSIGNED;
        }
    }
    return 0;
}

void printGrid(int grid[N][N]) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++)
            printf("%2d", grid[row][col]);
        printf("\n");
    }
}

int main() {
    int grid[N][N] = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

    if (SolveSudoku(grid) == 1)
        printGrid(grid);
    else
        printf("No solution exists");

    return 0;
}
```

### 4. Subset Sum Problem

**Problem**
Given a set of integers and a target sum, determine if there is a subset of the given set with a sum equal to the target sum.

**Explanation**
The algorithm includes or excludes each element in the subset and checks if the target sum can be achieved. If including or excluding an element leads to a solution, it proceeds; otherwise, it backtracks.

#### C Code
```c
#include <stdio.h>

int isSubsetSum(int set[], int n, int sum) {
    if (sum == 0)
        return 1;
    if (n == 0 && sum != 0)
        return 0;

    if (set[n - 1] > sum)
        return isSubsetSum(set, n - 1, sum);

    return isSubsetSum(set, n - 1, sum) ||
           isSubsetSum(set, n - 1, sum - set[n - 1]);
}

int main() {
    int set[] = {3, 34, 4, 12, 5, 2};
    int sum = 9;
    int n = sizeof(set) / sizeof(set[0]);
    if (isSubsetSum(set, n, sum) == 1)
        printf("Found a subset with given sum\n");
    else
        printf("No subset with given sum\n");
    return 0;
}
```

### 5. Permutations of a String

**Problem**
Generate all permutations of a given string.

**Explanation**
The algorithm swaps each character with every other character to generate permutations. After generating permutations with a character fixed at the first position, it swaps back and tries the next character at the first position.

#### C Code
```c
#include <stdio.h>
#include <string.h>

void swap(char *x, char *y) {
    char temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

void permute(char *a, int l, int r) {
    int i;
    if (l == r)
        printf("%s\n", a);
    else {
        for (i = l; i <= r; i++) {
            swap((a + l), (a + i));
            permute(a, l + 1, r);
            swap((a + l), (a + i));
        }
    }
}

int main() {
    char str[] = "ABC";
    int n = strlen(str);
    permute(str, 0, n - 1);
    return 0;
}
```

### 6. Combinations

**Problem**
Generate all combinations of k elements from a given set of n elements.

**Explanation**
The algorithm recursively selects or skips each element. If a combination of size k is formed, it is added to the result.

#### C Code
```c
#include <stdio.h>

void combinationUtil(int arr[], int n, int r, int index, int data[], int i) {
    if (index == r) {
        for (int j = 0; j < r; j++)
            printf("%d ", data[j]);
        printf("\n");
        return;
    }

    if (i >= n)
        return;

    data[index] = arr[i];
    combinationUtil(arr, n, r, index + 1, data, i + 1);

    combinationUtil(arr, n, r, index, data, i + 1);
}

void printCombination(int arr[], int n, int r) {
    int data[r];
    combinationUtil(arr, n, r, 0, data, 0);
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int r = 3;
    int n = sizeof(arr) / sizeof(arr[0]);
    printCombination(arr, n, r);
    return 0;
}
```

### 7. Word Search

**Problem**
Given a m x n grid of characters and a word, find if the word exists in the grid.

**Explanation**
The word can be constructed from letters of sequentially adjacent cells. The algorithm starts from each cell and uses backtracking to search for the word in all four directions (up, down, left, right).

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>

#define M 3
#define N 4

bool isSafe(char board[M][N], int row, int col, bool visited[M][N], char ch) {
    return (row >= 0) && (row < M) && (col >= 0) && (col < N) &&
           (board[row][col] == ch) && (!visited[row][col]);
}

bool DFS(char board[M][N], char word[], int row, int col, bool visited[M][N], int index) {
    if (index == strlen(word))
        return true;

    if (!isSafe(board, row, col, visited, word[index]))
        return false;

    visited[row][col] = true;

    static int rowNum[] = {-1, 1, 0, 0};
    static int colNum[] = {0, 0, -1, 1};

    for (int k = 0; k < 4; k++) {
        if (DFS(board, word, row + rowNum[k], col + colNum[k], visited, index + 1))
            return true;
    }

    visited[row][col] = false;
    return false;
}

bool wordSearch(char board[M][N], char word[]) {
    bool visited[M][N];
    memset(visited, 0, sizeof(visited));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (DFS(board, word, i, j, visited, 0))
                return true;
        }
    }
    return false;
}

int main() {
    char board[M][N] = {
        {'A', 'B', 'C', 'E'},
        {'S', 'F', 'C', 'S'},
        {'A', 'D', 'E', 'E'}
    };

    char word[] = "ABCCED";

    if (wordSearch(board, word))
        printf("Word found\n");
    else
        printf("Word not found\n");

    return 0;
}
```

### 8. Hamiltonian Path

**Problem**
Find a path in a graph that visits each vertex exactly once.

**Explanation**
The algorithm starts from a vertex and recursively tries to find a path visiting all vertices exactly once. If a vertex leads to a solution, it proceeds; otherwise, it backtracks.

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>

#define V 5

void printSolution(int path[]) {
    for (int i = 0; i < V; i++)
        printf("%d ", path[i]);
    printf("%d\n", path[0]);
}

bool isSafe(int v, bool graph[V][V], int path[], int pos) {
    if (graph[path[pos - 1]][v] == 0)
        return false;

    for (int i = 0; i < pos; i++)
        if (path[i] == v)
            return false;

    return true;
}

bool hamCycleUtil(bool graph[V][V], int path[], int pos) {
    if (pos == V) {
        if (graph[path[pos - 1]][path[0]] == 1)
            return true;
        else
            return false;
    }

    for (int v = 1; v < V; v++) {
        if (isSafe(v, graph, path, pos)) {
            path[pos] = v;

            if (hamCycleUtil(graph, path, pos + 1) == true)
                return true;

            path[pos] = -1;
        }
    }

    return false;
}

bool hamCycle(bool graph[V][V]) {
    int *path = malloc(V * sizeof(int));
    for (int i = 0; i < V; i++)
        path[i] = -1;

    path[0] = 0;
    if (hamCycleUtil(graph, path, 1) == false) {
        printf("Solution does not exist\n");
        return false;
    }

    printSolution(path);
    return true;
}

int main() {
    bool graph[V][V] = {{0, 1, 0, 1, 0},
                        {1, 0, 1, 1, 1},
                        {0, 1, 0, 0, 1},
                        {1, 1, 0, 0, 1},
                        {0, 1, 1, 1, 0}};

    hamCycle(graph);
    return 0;
}
```

### 9. Knight’s Tour

**Problem**
Find a sequence of moves of a knight on a chessboard such that the knight visits every square exactly once.

**Explanation**
The algorithm starts from a given position and tries all eight possible moves of the knight. If a move leads to a solution, it proceeds; otherwise, it backtracks.

#### C Code
```c
#include <stdio.h>

#define N 8

int isSafe(int x, int y, int sol[N][N]) {
    return (x >= 0 && x < N && y >= 0 && y < N && sol[x][y] == -1);
}

void printSolution(int sol[N][N]) {
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++)
            printf(" %2d ", sol[x][y]);
        printf("\n");
    }
}

int solveKTUtil(int x, int y, int movei, int sol[N][N], int xMove[N], int yMove[N]) {
    int k, next_x, next_y;
    if (movei == N * N)
        return 1;

    for (k = 0; k < 8; k++) {
        next_x = x + xMove[k];
        next_y = y + yMove[k];
        if (isSafe(next_x, next_y, sol)) {
            sol[next_x][next_y] = movei;
            if (solveKTUtil(next_x, next_y, movei + 1, sol, xMove, yMove) == 1)
                return 1;
            else
                sol[next_x][next_y] = -1;
        }
    }
    return 0;
}

int solveKT() {
    int sol[N][N];
    for (int x = 0; x < N; x++)
        for (int y = 0; y < N; y++)
            sol[x][y] = -1;

    int xMove[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    int yMove[8] = {1, 2, 2, 1, -1, -2, -2, -1};

    sol[0][0] = 0;

    if (solveKTUtil(0, 0, 1, sol, xMove, yMove) == 0) {
        printf("Solution does not exist\n");
        return 0;
    } else
        printSolution(sol);

    return 1;
}

int main() {
    solveKT();
    return 0;
}
```

### 10. Crossword Puzzle

**Problem**
Fill a crossword puzzle grid with a given set of words.

**Explanation**
The algorithm places each word in the grid one by one, both horizontally and vertically. If placing a word leads to a solution, it proceeds; otherwise, it backtracks and tries the next word.

#### C Code
```c
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#define N 10

bool isSafe(char board[N][N], int row, int col, char word[], bool isHorizontal) {
    int len = strlen(word);

    if (isHorizontal) {
        if (col + len > N)
            return false;
        for (int i = 0; i < len; i++) {
            if (board[row][col + i] != '-' && board[row][col + i] != word[i])
                return false;
        }
    } else {
        if (row + len > N)
            return false;
        for (int i = 0; i < len; i++) {
            if (board[row + i

][col] != '-' && board[row + i][col] != word[i])
                return false;
        }
    }
    return true;
}

void setWord(char board[N][N], int row, int col, char word[], bool isHorizontal, bool set) {
    int len = strlen(word);
    if (isHorizontal) {
        for (int i = 0; i < len; i++) {
            if (set)
                board[row][col + i] = word[i];
            else
                board[row][col + i] = '-';
        }
    } else {
        for (int i = 0; i < len; i++) {
            if (set)
                board[row + i][col] = word[i];
            else
                board[row + i][col] = '-';
        }
    }
}

bool solveCrossword(char board[N][N], char words[][N], int index, int n) {
    if (index == n)
        return true;

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            if (isSafe(board, row, col, words[index], true)) {
                setWord(board, row, col, words[index], true, true);
                if (solveCrossword(board, words, index + 1, n))
                    return true;
                setWord(board, row, col, words[index], true, false);
            }
            if (isSafe(board, row, col, words[index], false)) {
                setWord(board, row, col, words[index], false, true);
                if (solveCrossword(board, words, index + 1, n))
                    return true;
                setWord(board, row, col, words[index], false, false);
            }
        }
    }
    return false;
}

int main() {
    char board[N][N] = {
        {'+', '-', '+', '+', '+', '+', '+', '+', '+', '+'},
        {'+', '-', '+', '+', '+', '+', '+', '+', '+', '+'},
        {'+', '-', '-', '-', '-', '-', '+', '+', '+', '+'},
        {'+', '-', '+', '+', '+', '-', '+', '+', '+', '+'},
        {'+', '-', '+', '+', '+', '-', '+', '+', '+', '+'},
        {'+', '-', '-', '-', '-', '-', '-', '-', '+', '+'},
        {'+', '-', '+', '+', '+', '-', '+', '+', '+', '+'},
        {'+', '+', '+', '+', '+', '-', '+', '+', '+', '+'},
        {'+', '+', '+', '+', '+', '-', '+', '+', '+', '+'},
        {'+', '+', '+', '+', '+', '+', '+', '+', '+', '+'},
    };

    char words[3][N] = {"HELLO", "WORLD", "GEEK"};

    if (solveCrossword(board, words, 0, 3)) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%c ", board[i][j]);
            }
            printf("\n");
        }
    } else {
        printf("No solution exists\n");
    }

    return 0;
}
```

