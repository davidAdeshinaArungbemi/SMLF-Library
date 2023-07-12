#ifndef DATAFRAME_DESC
#define DATAFRAME_DESC
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <cmath>
namespace MLA
{

	class DataFrame
	{
	private:
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Data;
		std::map<std::string, size_t> feature_map;
		std::vector<std::string> feature_list;

		void acquireFeatureNames(std::string line);
		Eigen::VectorXf vectoriseRows(std::string line, const size_t cols);

		void readData(std::string file_path);

		void makeIdentity(size_t num_rows, size_t num_cols);

		void randomFill(size_t num_rows, size_t num_cols, float min, float max);

		void constantFill(size_t num_rows, size_t num_cols, float val);

	public:
#define IDENTITY 0
#define RANDOM 1
#define CONSTANT 2

		// #define ROW_END (this->Data).rows()
		// #define COL_END (this->Data).cols()

		DataFrame() = default;
		DataFrame(std::string file_path);
		DataFrame(const size_t fillType, size_t num_rows, size_t num_cols,
				  size_t a = 0, size_t b = 1);
		DataFrame(size_t num_rows, size_t num_cols, Eigen::VectorXf element_vec);
		DataFrame(Eigen::MatrixXf Data);

		size_t rowSize() const;
		size_t colSize() const;
		size_t size() const;
		size_t maximum() const;
		size_t minimum() const;

		void makeInverse();
		void makeTranspose();
		void makeAdjoint();

		DataFrame viewInverse();
		DataFrame viewTranspose();
		DataFrame viewAdjoint();
		DataFrame featureView(std::vector<std::string> featureVec);
		DataFrame locView(size_t row_1, size_t row_2, size_t col_1, size_t col_2);
		DataFrame head(size_t num_rows);
		DataFrame dropColumnView(std::vector<size_t> column_vec);
		friend DataFrame operator*(const MLA::DataFrame dataFrame, const MLA::DataFrame &other); // overloading multiplication operator, friend keyword is so cool
		//	friend allows me to grant access of private and protected attributes to external functions or classes

		friend std::ostream &operator<<(std::ostream &os, const DataFrame &dataFrame);

		//	const Eigen::MatrixXf getData() const;

		void printFeatures();
		void createCSV(std::string fileName, std::string directory = "");
		void swapColumns(size_t col_1, size_t col_2);
		void printShape();

		std::vector<float> convertToVector();
	};
}
#endif

#ifndef DataFRAME_IMPLEMENTATION
#define DataFRAME_IMPLEMENTATION
#include "DataFrame.hpp"

// private
void MLA::DataFrame::acquireFeatureNames(std::string line)
{
	size_t start_loc = 0;
	size_t end_loc;
	size_t size; // size of substring
	std::string substring;

	line += ","; // appended a comma to solve the issue of selecting last number in string
	size_t feature_loc = 0;
	for (size_t i = 0; i < line.size(); i++)
	{
		if (line[i] == ',')
		{
			end_loc = i;

			size = (end_loc - start_loc);
			substring = line.substr(start_loc, size);
			start_loc = end_loc + 1;
			this->feature_map[substring] = feature_loc;
			this->feature_list.push_back(substring);
			feature_loc += 1;
		}
	}
}

Eigen::VectorXf MLA::DataFrame::vectoriseRows(std::string line,
											  const size_t cols)
{
	size_t start_loc = 0;
	size_t end_loc;
	size_t size; // size of substring
	std::string substring;
	size_t index = 0;
	Eigen::RowVectorXf row_vector(cols);

	line += ","; // appended a comma to solve the issue of selecting last number in string

	for (size_t i = 0; i < line.size(); i++)
	{
		if (line[i] == ',')
		{
			end_loc = i;

			size = (end_loc - start_loc);
			substring = line.substr(start_loc, size);
			start_loc = end_loc + 1;

			row_vector(index) = stof(substring);
			index++;
		}
	}
	return row_vector;
}

void MLA::DataFrame::readData(std::string file_path)
{
	std::ifstream csv_file;
	csv_file.open(file_path);

	if (csv_file)
	{
		std::string line;
		getline(csv_file, line);
		acquireFeatureNames(line);

		size_t cols = std::count(line.begin(), line.end(), ',') + 1;
		size_t rows = 0;
		this->Data.conservativeResize(rows, cols);
		while (getline(csv_file, line))
		{
			rows++;
			this->Data.conservativeResize(rows, cols);
			Eigen::RowVectorXf row_vec = vectoriseRows(line, cols);
			this->Data.bottomRows(row_vec.rows()) = row_vec;
		}
		csv_file.close();
		return; // if successful end here
	}

	std::cerr << "File not found!" << std::endl;
	assert(false);
	return;
}

void MLA::DataFrame::makeIdentity(size_t num_rows, size_t num_cols)
{
	this->Data.conservativeResize(num_rows, num_cols);
	this->Data.setIdentity();
}

void MLA::DataFrame::randomFill(size_t num_rows, size_t num_cols, float min,
								float max)
{
	srand((unsigned int)time(0));
	this->Data.conservativeResize(num_rows, num_cols);
	this->Data.setRandom();

	for (size_t pos = 0; pos < num_rows * num_cols; pos++)
	{
		size_t i = pos / num_cols;
		size_t j = pos % num_cols;

		this->Data(i, j) = (this->Data(i, j) + 1) * (max - min) / (1 + 1) + min;
	}
}

void MLA::DataFrame::constantFill(size_t num_rows, size_t num_cols,
								  float val)
{
	this->Data.conservativeResize(num_rows, num_cols);
	this->Data.setConstant(val);
}

// public
MLA::DataFrame::DataFrame(const std::string file_path)
{
	readData(file_path);
}

MLA::DataFrame::DataFrame(const size_t fillType, size_t num_rows,
						  size_t num_cols, size_t a, size_t b)
{
	switch (fillType)
	{
	case 0:
		makeIdentity(num_rows, num_cols);
		break;
	case 1:
		randomFill(num_rows, num_cols, a, b);
		break;
	case 2:
		constantFill(num_rows, num_cols, a);
		break;
	}
}

MLA::DataFrame::DataFrame(size_t num_rows, size_t num_cols,
						  Eigen::VectorXf element_vec)
{
	assert(num_cols * num_rows == element_vec.size());
	this->Data.conservativeResize(num_rows, num_cols);
	this->Data << element_vec;
}

MLA::DataFrame::DataFrame(Eigen::MatrixXf data)
{
	this->Data = data;
}

size_t MLA::DataFrame::rowSize() const
{
	return this->Data.rows();
}

size_t MLA::DataFrame::colSize() const
{
	return this->Data.cols();
}

size_t MLA::DataFrame::size() const
{
	return this->Data.size();
}

size_t MLA::DataFrame::maximum() const
{
	return this->Data.maxCoeff();
}
size_t MLA::DataFrame::minimum() const
{
	return this->Data.minCoeff();
}

void MLA::DataFrame::printShape()
{
	printf("Shape: %zu, %zu\n", this->Data.rows(), this->Data.cols());
}

void MLA::DataFrame::makeInverse()
{
	this->Data.reverseInPlace();
}

MLA::DataFrame MLA::DataFrame::viewInverse()
{
	return DataFrame(this->Data.inverse());
}

void MLA::DataFrame::makeTranspose()
{
	this->Data.transposeInPlace();
}

MLA::DataFrame MLA::DataFrame::viewTranspose()
{
	return DataFrame(this->Data.transpose());
}

void MLA::DataFrame::makeAdjoint()
{
	this->Data.adjointInPlace();
}

MLA::DataFrame MLA::DataFrame::viewAdjoint()
{
	return DataFrame(this->Data.adjoint());
}

MLA::DataFrame MLA::DataFrame::featureView(
	std::vector<std::string> featureVec)
{
	Eigen::MatrixXf dataView(this->Data.rows(), featureVec.size());
	size_t new_j = 0;
	for (auto feature : featureVec)
	{
		size_t j = feature_map[feature];

		for (size_t i = 0; i < this->Data.rows(); i++)
		{
			dataView(i, new_j) = this->Data(i, j);
		}

		new_j++;
	}

	return DataFrame(dataView);
}

MLA::DataFrame MLA::DataFrame::locView(size_t row_1, size_t row_2, size_t col_1,
									   size_t col_2)
{

	assert(row_1 <= row_2);
	assert(col_1 <= col_2);
	Eigen::MatrixXf dataView(row_2 - row_1 + 1, col_2 - col_1 + 1);

	size_t pos = 0;

	for (size_t i = 0; i < this->Data.rows(); i++)
	{
		for (size_t j = 0; j < this->Data.cols(); j++)
		{
			if (i >= row_1 && i <= row_2 && j >= col_1 && j <= col_2)
			{
				size_t k, l; // dataView row and column
				k = pos / dataView.cols();
				l = pos % dataView.cols();

				dataView(k, l) = this->Data(i, j);
				pos++;
			}
		}
	}
	return DataFrame(dataView);
}

MLA::DataFrame MLA::operator*(const MLA::DataFrame dataFrame,
							  const MLA::DataFrame &other)
{
	return MLA::DataFrame(dataFrame.Data * other.Data); // @suppress("Symbol is not resolved")
}

MLA::DataFrame MLA::DataFrame::head(size_t num_rows)
{
	assert(num_rows <= this->Data.rows());
	return locView(0, num_rows - 1, 0, this->Data.cols() - 1);
}

std::ostream &MLA::operator<<(std::ostream &os, const DataFrame &dataFrame)
{
	size_t max_digit = 1;

	for (size_t i = 0; i < dataFrame.rowSize(); i++)
	{ // find maximum number of digit
		for (size_t j = 0; j < dataFrame.colSize(); j++)
		{
			size_t digits = std::to_string(dataFrame.Data(i, j)).length();
			if (digits > max_digit)
			{
				max_digit = digits;
			}
		}
	}

	if (max_digit < std::to_string(dataFrame.colSize()).length())
	{ // in case header length is greater than length of maximum digit
		max_digit = dataFrame.colSize() + 1;
	}

	size_t index_spacing = std::to_string(dataFrame.rowSize()).length(); // if there's some error with row loc spacing check here

	size_t h_length = ((max_digit + 3) * dataFrame.colSize()) + index_spacing + 2; // decide number of dashes
	std::string h_dash(h_length, '-');

	// feature header begin
	std::cout << h_dash << std::endl;
	std::cout << std::setw(index_spacing + 2) << "|";
	for (size_t j = 0; j < dataFrame.colSize(); j++)
	{
		size_t diff = std::to_string(j).length();
		std::cout << std::setw(max_digit + 2 - diff) << "F" << j << "|";
	}
	std::cout << "\n";
	std::cout << h_dash << std::endl;
	// feature header end

	// feature begin
	for (size_t i = 0; i < dataFrame.rowSize(); i++)
	{
		std::cout << std::setw(index_spacing + 1) << i << "|";
		for (size_t j = 0; j < dataFrame.colSize(); j++)
		{
			std::cout << std::setw(max_digit + 2) << dataFrame.Data(i, j) << "|";
		}
		std::cout << std::endl;
		std::cout << h_dash << std::endl;
	}
	// feature end

	os << "Shape: "
	   << "(" << dataFrame.rowSize() << ", " << dataFrame.colSize()
	   << ")";
	std::cout << "\n";
	return os;
}

void MLA::DataFrame::printFeatures()
{
	size_t i = 0;
	for (auto feature : this->feature_list)
	{
		printf("[%zu] %s\n", i, feature.c_str());
		i++;
	}
}

std::vector<float> MLA::DataFrame::convertToVector()
{
	std::vector<float> stdVector(Data.data(), Data.data() + Data.size());
	return stdVector;
}

void MLA::DataFrame::createCSV(std::string fileName, std::string directory)
{
	std::ofstream file;
	if (directory != "")
	{
		file.open(directory + "/" + fileName + ".csv");
	}
	else
	{
		file.open(fileName + ".csv");
	}

	if (file.is_open())
	{
		// Customising the output format
		Eigen::IOFormat csvFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, // @suppress("Invalid arguments")
								  ", ", "\n");

		// Saving the matrix to the CSV file
		file << this->Data.format(csvFormat);

		file.close();
		std::cout << "CSV file save successful." << std::endl;
	}
	else
	{
		std::cerr << "Unable to open the file." << std::endl;
	}
}

void MLA::DataFrame::swapColumns(size_t col_1, size_t col_2)
{
	for (size_t i = 0; i < this->Data.rows(); i++)
	{
		float temp = this->Data(i, col_1);
		this->Data(i, col_1) = this->Data(i, col_2);
		this->Data(i, col_2) = temp;
	}
}

MLA::DataFrame MLA::DataFrame::dropColumnView(std::vector<size_t> column_vec)
{
	assert(column_vec.size() < this->Data.cols());
	size_t new_cols = this->Data.cols() - column_vec.size();

	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> reduced_data(
		this->Data.rows(), new_cols);

	size_t new_col_index = 0;

	for (size_t j = 0; j < this->Data.cols(); j++)
	{
		if (std::find(column_vec.begin(), column_vec.end(), j) == column_vec.end())
		{ // check if j among dropped columns, if not then
			for (size_t i = 0; i < this->Data.rows(); i++)
			{
				reduced_data(i, new_col_index) = this->Data(i, j);
			}
			new_col_index++;
		}
	}

	return DataFrame(reduced_data);
}

#endif
