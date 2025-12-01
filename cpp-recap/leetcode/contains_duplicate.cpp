#include <iostream>
#include <vector>
#include <unordered_set>
using namespace std;

/*class Solution {
public:
    bool hasDuplicate(std::vector<int>& nums) {
        return true;
    }
};*/

void print(const vector<int>& nums) {
    //for (int i=0; i < nums.size(); i++) {
    //    std::cout << nums[i] << " ";
    //}

    // for (auto elem : il) will create a temporary copy of each element. Usually less efficient.

    // for (auto& elem: il) won't create copies, and allow you to modify the elements if the underlying container allows that (e.g. not const). But it cannot bind to rvalue references, so it cannot be used on containers that return proxy objects (e.g. std::vector<bool>).

    // for (const auto& elem: il) won't create copies either, and won't allow you to do any modifications. This can prevent accidental modifications and signify your intent clearly.

    for (const auto& i : nums) {
        cout << nums[i] << " ";
    }
    cout << std::endl;
}

bool hasDuplicate(vector<int>& nums) {
    unordered_set<int> set;
    bool result = false;
    for (const int& i : nums) {
        if (set.count(i)) {
            result = true;
            break;
        }
        set.insert(i);
    }
    return result;
}

int main() {
    std::vector<int> test1 = {1, 2, 3, 4, 5};
    std::vector<int> test2 = {1, 2, 3, 5, 5};
    std::vector<int> test3 = {1};
    std::vector<int> test4 = {};
    cout << hasDuplicate(test1) << endl;
    cout << hasDuplicate(test2) << endl;
    cout << hasDuplicate(test3) << endl;
    cout << hasDuplicate(test4) << endl;
    return 0;
}