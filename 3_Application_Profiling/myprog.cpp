#include <iostream>
#include <vector>

void quicksort(std::vector<int> &arr, int low, int high);
int partition(std::vector<int> &arr, int low, int high);
void swap(int &a, int &b);

int main()
{
    std::vector<int> arr = {10, 7, 8, 9, 1, 5};

    std::cout << "Unsorted array: ";
    for (int i : arr)
        std::cout << i << " ";
    std::cout << std::endl;

    quicksort(arr, 0, arr.size() - 1);

    std::cout << "Sorted array: ";
    for (int i : arr)
        std::cout << i << " ";
    std::cout << std::endl;

    return 0;
}

void quicksort(std::vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);

        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int partition(std::vector<int> &arr, int low, int high)
{
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}
