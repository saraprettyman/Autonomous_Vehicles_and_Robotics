#include <iostream>
#include <iomanip>
#include <cmath>

template <typename DataType>
struct Point
{
    DataType x_mm, y_mm;  // x and y are in millimeters
};

template <typename DataType>
DataType ComputeAreaOfTriangle(
    const Point<DataType> first_point,
    const Point<DataType> second_point,
    const Point<DataType> third_point)
{
    // Compute the three side lengths from the points
    const auto a_mm = std::hypot(first_point.x_mm - second_point.x_mm, first_point.y_mm - second_point.y_mm);
    const auto b_mm = std::hypot(second_point.x_mm - third_point.x_mm, second_point.y_mm - third_point.y_mm);
    const auto c_mm = std::hypot(third_point.x_mm - first_point.x_mm, third_point.y_mm - first_point.y_mm);

    // Compute the area of the triangle
    const auto s_mm = (a_mm + b_mm + c_mm) / 2;
    const auto area_mm_sq = std::sqrt(s_mm * (s_mm - a_mm) * (s_mm - b_mm) * (s_mm - c_mm));

    // Return the area
    return area_mm_sq;
}

int main()
{
    using Type = double;  // Using double precision floating point for higher accuracy

    Point<Type> pt_a, pt_b, pt_c;

    // Point A (x, y) coordinates at the SW corner of Utah
    pt_a.x_mm = 0.0; 
    pt_a.y_mm = 0.0;

    // Point B (x, y) coordinates at the SE corner of Utah
    pt_b.x_mm = 435000000.0; // 435 km converted to millimeters
    pt_b.y_mm = 0.0; 

    // Point C (x, y) coordinates at the NW corner of Utah
    pt_c.x_mm = 0.0;
    pt_c.y_mm = 563000000.0; // 563 km converted to millimeters

    // Set the console to high precision
    std::cout << std::setprecision(40);

    // Print out the data to console
    std::cout << "The area of the triangle bound by points:\n" <<
       "\t(" << pt_a.x_mm << "," << pt_a.y_mm << ")\n" <<
       "\t(" << pt_b.x_mm << "," << pt_b.y_mm << ")\n" <<
       "\t(" << pt_c.x_mm << "," << pt_c.y_mm << ")\n" <<
       "is " << ComputeAreaOfTriangle<Type>(pt_a, pt_b, pt_c) << " square millimeters." << std::endl; 


    return 0;
}