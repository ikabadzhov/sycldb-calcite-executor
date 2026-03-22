#include "ssb_utils.h"
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>

using namespace std;

int main(int argc, char** argv) {
  if (argc != 4) {
    cout << "col-name len SF" << endl;
    return 1;
  }

  string col_name = argv[1];
  int len = atoi(argv[2]);
  string sf = argv[3];

  cout << "Calculating minmax for " << col_name << " (len=" << len << ") SEGMENT_SIZE=" << SEGMENT_SIZE << endl;

  uint *raw = loadColumn<uint>(col_name, len);
  if (raw == NULL) {
      cerr << "ERROR: Failed to load column " << col_name << endl;
      return 1;
  }

  cout << "Loaded Column " << col_name << endl;

  ofstream myfile;
  myfile.open (string(DATA_DIR) + col_name + "minmax");
  if (!myfile.is_open()) {
      cerr << "ERROR: Failed to open output file for " << col_name << endl;
      return 1;
  }

  size_t total_segment = ((size_t)len + SEGMENT_SIZE - 1)/SEGMENT_SIZE;

  for (size_t i = 0; i < total_segment; i++) {
  	size_t adjusted_len = SEGMENT_SIZE;
  	if (i == total_segment-1) {
  		adjusted_len = (size_t)len - (size_t)SEGMENT_SIZE * i;
  	}

    uint min = raw[i*(size_t)SEGMENT_SIZE];
    uint max = raw[i*(size_t)SEGMENT_SIZE];
  	for (size_t j = 0; j < adjusted_len; j++) {
  		if (raw[i*(size_t)SEGMENT_SIZE + j] > max) max = raw[i*(size_t)SEGMENT_SIZE + j];
  		if (raw[i*(size_t)SEGMENT_SIZE + j] < min) min = raw[i*(size_t)SEGMENT_SIZE + j];
  	}
  	myfile << min << " " << max << '\n';
  }

  myfile.close();
  cout << "Finished " << col_name << endl;

  return 0;
}