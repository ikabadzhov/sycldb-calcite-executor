#include <algorithm>
#include <string>
#include <fstream>
#include <sycl/sycl.hpp>

#pragma once

using namespace std;

#define DATA_DIR "/tmp/data/s20_columnar/"

template <typename T>
struct ColumnData
{
    T *content;
    bool has_ownership;       // Indicates if this column owns the memory for its content
    bool is_aggregate_result; // Indicates if this column is the result of an aggregate operation
    T min_value;              // Minimum value in the column
    T max_value;              // Maximum value in the column
};

template <typename T>
struct TableData
{
    ColumnData<T> *columns;
    int columns_size;    // length of the columns array (number of columns loaded)
    int col_len;         // number of rows
    int col_number;      // total number of columns in the table
    bool *flags;         // selection flags
    int group_by_column; // column number used for grouping, -1 if not used
    std::string table_name;
    std::map<int, int> column_indices; // Maps column numbers from calcite to its index in the columns array
};

template <typename T>
T *loadColumn(string table_name, int col_index, int& table_len, sycl::queue &queue) {
  std::transform(table_name.begin(), table_name.end(), table_name.begin(), ::toupper);
  string col_name = table_name + std::to_string(col_index);
  string filename = DATA_DIR + col_name;
  std::cout << "Loading column: " << filename << std::endl;
  ifstream colData(filename.c_str(), ios::in | ios::binary);
  if (!colData) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return NULL;
  }
  colData.seekg(0, std::ios::end);
  std::streampos fileSize = colData.tellg();
  int num_entries = static_cast<int>(fileSize / sizeof(T));
  colData.seekg(0, std::ios::beg);
  //T *h_col = new T[num_entries];
  //colData.read((char *)h_col, num_entries * sizeof(T));
  T *h_col = sycl::malloc_host<T>(num_entries, queue);
  colData.read((char *)h_col, num_entries * sizeof(T));
  T *d_col = sycl::malloc_device<T>(num_entries, queue);
  queue.memcpy(d_col, h_col, num_entries * sizeof(T)).wait();
  sycl::free(h_col, queue);
  queue.wait();
  //queue.memcpy(h_col2, h_col, num_entries * sizeof(T)).wait();
  table_len = num_entries;
  return d_col;
}

TableData<int> loadTable(std::string table_name, int col_number, const std::set<int> &columns, sycl::queue &queue)
{
    TableData<int> res;

    res.col_number = col_number;
    res.columns_size = col_number; // in dummy we load all columns
    res.table_name = table_name;

    res.columns = new ColumnData<int>[col_number];



    for (auto &col_idx : columns)
    {
        res.column_indices[col_idx] = col_idx; // map the column index to itself
        //res.columns[col_idx].content = loadColumn<int>(res.table_name, col_idx, res.col_len, queue);
        auto table_name = res.table_name;
        std::transform(table_name.begin(), table_name.end(), table_name.begin(), ::toupper);
        string col_name = table_name + std::to_string(col_idx);
        string filename = DATA_DIR + col_name;
        std::cout << "Loading column: " << filename << std::endl;
        ifstream colData(filename.c_str(), ios::in | ios::binary);
        colData.seekg(0, std::ios::end);
        std::streampos fileSize = colData.tellg();
        int num_entries = static_cast<int>(fileSize / sizeof(int));
        colData.seekg(0, std::ios::beg);
        int *h_col = sycl::malloc_host<int>(num_entries, queue);
        colData.read((char *)h_col, num_entries * sizeof(int));
        res.columns[col_idx].content = sycl::malloc_device<int>(num_entries, queue);
        queue.memcpy(res.columns[col_idx].content, h_col, num_entries * sizeof(int)).wait();
        queue.wait();
        res.col_len = num_entries;
        res.columns[col_idx].has_ownership = true;
        res.columns[col_idx].is_aggregate_result = false;
        //res.col_len = sizeof(res.columns[col_idx].content) / sizeof(int);
        res.columns[col_idx].min_value = *std::min_element(h_col, h_col + res.col_len);
        res.columns[col_idx].max_value = *std::max_element(h_col, h_col + res.col_len);
        sycl::free(h_col, queue);
    }

    std::cout << "Loaded table: " << res.table_name << " with " << res.col_len << " rows and " << res.col_number << " columns." << std::endl;
    bool *flags = new bool[res.col_len];
    std::fill_n(flags, res.col_len, true);
    res.flags = sycl::malloc_shared<bool>(res.col_len, queue);
    queue.prefetch(res.flags, res.col_len * sizeof(bool));
    queue.memcpy(res.flags, flags, res.col_len * sizeof(bool)).wait();
    return res;
}