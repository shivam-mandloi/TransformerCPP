#pragma once

#include <iostream>
#include <cstddef>


template <typename T>
class vex
{
public:
    T *arr;
    int len, row, col;
    bool isTransposed = false;
    vex():row(0), col(0), len(0){}
    vex(size_t _row, size_t _col, T defaultData)
    {
        row = _row;
        col = _col;
        arr = new T[row * col];
        len = row * col;
        std::fill(arr, arr + len, defaultData);
    }
    // vex(size_t _row, size_t _col) : vex(_row, _col, T{}) {}
    vex(size_t _col, T defaultData) : vex(1, _col, defaultData) {}
    vex(size_t _col) : vex(1, _col, T{}) {}    
    
    vex(const vex<T> &newArr): row(newArr.row), col(newArr.col), len(newArr.len)
    {
        arr = new T[len];
        std::copy(newArr.arr, newArr.arr + newArr.len, arr);
    }

    vex(std::initializer_list<T> list) : row(1), col(list.size()), len(list.size())
    {
        arr = new T[len];
        std::copy(list.begin(), list.end(), arr);
    }

    vex<T>& operator=(const vex<T> &other)
    {
        if(this != &other)
        {
            delete[] arr;
            len = other.len;
            row = other.row;
            col = other.col;
            arr = new T[len];
            std::copy(other.arr, other.arr + len, arr);            
        }
        return *this;
    }

    ~vex()
    {        
        delete[] arr;
    }

    T Get(int x)
    {
        return arr[x];
    }

    T Get(int i, int j)
    {
        return isTransposed ? Get(row * j + i) : Get(col * i + j);
    }

    void push(int i, T data)
    {
        arr[i] = data;
    }

    void push(int i, int j, T data)
    {
        if(isTransposed)
            push(j * row + i, data);
        else
            push(i * col + j, data);
    }

    vex<int> size()
    {
        vex<int> size(2, 0);
        size.push(0, row);
        size.push(1, col);
        return size;
    }

    void TR()
    {
        isTransposed = !isTransposed;
        int temp = row;
        row = col;
        col = temp;
    }

    void print()
    {
        try
        {
            for(int i = 0; i < row; i++)
            {            
                for(int j = 0; j < col; j++)
                {
                    std::cout << Get(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }
};

