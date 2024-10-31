#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    // Load the image
    cv::Mat image = cv::imread("/Users/navi/Desktop/open_cv/Assignment_1/tables/Table1.jpg");
    if (image.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat gray, blurred, edges;

    // Convert to grayscale
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Optional: Apply blur
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Detect edges
    cv::Canny(blurred, edges, 50, 150);

    // Use morphological close to close gaps in the edges (optional)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Draw contours with an area greater than 100
    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area > 1000000)
        {
            cv::drawContours(image, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 255, 0), 3);
        }
    }

    // Display the result
    cv::namedWindow("Detected Table", cv::WINDOW_AUTOSIZE);
    cv::imshow("Detected Table", image);
    cv::waitKey(0);

    return 0;
}
