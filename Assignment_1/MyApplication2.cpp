#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Function to check if two lines intersect
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2,
                  cv::Point2f &r)
{
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
    cv::Point2f d2 = p2 - o2;

    float cross = d1.x * d2.y - d1.y * d2.x;
    if (std::abs(cross) < 1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x) / cross;
    r = o1 + d1 * t1;
    return true;
}

int main()
{
    // Load the image
    cv::Mat src = cv::imread("/Users/navi/Desktop/open_cv/Assignment_1/tables/Table1.jpg");

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian Blur
    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 0);

    // Canny edge detector
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150, 3);

    // Hough line transform
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 150);

    // Find the corners by intersecting lines
    std::vector<cv::Point2f> corners;
    for (size_t i = 0; i < lines.size(); i++)
    {
        for (size_t j = i + 1; j < lines.size(); j++)
        {
            cv::Vec2f line1 = lines[i];
            cv::Vec2f line2 = lines[j];
            float rho1 = line1[0], theta1 = line1[1];
            float rho2 = line2[0], theta2 = line2[1];

            cv::Point2f pt1, pt2;
            pt1.x = cos(theta1) * rho1;
            pt1.y = sin(theta1) * rho1;
            pt2.x = cos(theta2) * rho2;
            pt2.y = sin(theta2) * rho2;

            cv::Point2f intersection;
            if (intersection(pt1, pt1 + cv::Point2f(-sin(theta1), cos(theta1)),
                             pt2, pt2 + cv::Point2f(-sin(theta2), cos(theta2)), intersection))
            {
                if (intersection.inside(cv::Rect(0, 0, src.cols, src.rows)))
                {
                    corners.push_back(intersection);
                }
            }
        }
    }

    // Draw the corners
    for (auto &corner : corners)
    {
        cv::circle(src, corner, 5, cv::Scalar(0, 255, 0), -1);
    }

    // Display the results
    cv::imshow("Detected Lines (in red) - Corners (in green)", src);
    cv::waitKey();

    return 0;
}
