#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

float colorDistance(const Vec3f &color1, const Vec3f &color2)
{
    return norm(color1 - color2);
}

int main()
{
    // Only one filename since we are detecting the table in one image.
    string filename = "/Users/navi/Desktop/open_cv/Assignment_1/tables/Table4.jpg";

    Mat src = imread(filename, IMREAD_COLOR);

    // Ensure the image was loaded properly
    if (src.empty())
    {
        cerr << "Error: Image could not be read." << endl;
        return -1;
    }
    Mat display = src.clone();

    // Apply Gaussian Blur
    GaussianBlur(src, src, Size(7, 7), 1, 1);

    // Convert image to float type for kmeans
    Mat data;
    src.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // Use KMeans to cluster the image colors into K clusters
    Mat labels, centers;
    int K = 30; // Assuming two dominant colors: blue for the table and others
    kmeans(data, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // Here you would define the blue color of your table.
    Vec3f blueTableColor(135, 100, 60); // Define the BGR values of the specific shade of blue for your table.

    vector<int> relevantClusters;

    // Find the cluster index that corresponds to the table's color
    for (int k = 0; k < K; k++)
    {
        Vec3f clusterColor(centers.at<float>(k, 0), centers.at<float>(k, 1), centers.at<float>(k, 2));
        if (colorDistance(clusterColor, blueTableColor) < 70.0f)
        { // Threshold for color matching, adjust as needed
            relevantClusters.push_back(k);
        }
    }

    Mat binImg = Mat::zeros(src.size(), CV_8U);
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            int clusterIdx = labels.at<int>(y * src.cols + x);
            if (find(relevantClusters.begin(), relevantClusters.end(), clusterIdx) != relevantClusters.end())
            {
                binImg.at<uchar>(y, x) = 255; // Mark the relevant clusters as white
            }
        }
    }
    imshow("bnw", binImg);
    waitKey(0);

    // Morphological operations
    cv::Mat morphed;
    cv::dilate(binImg, binImg, cv::Mat(), cv::Point(-1, -1), 2); // Adjust iterations as necessary
    imshow("morphed", binImg);
    waitKey(0);

    // Find contours
    vector<vector<Point>> contours;
    findContours(binImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Analyze contours
    vector<vector<Point>> detectedTables;
    for (const auto &contour : contours)
    {
        double epsilon = 0.02 * arcLength(contour, true); // Approximation accuracy
        vector<Point> approx;
        approxPolyDP(contour, approx, epsilon, true);

        if (approx.size() == 4)
        {
            double rectArea = contourArea(approx);
            double contArea = contourArea(contour);

            // Check the rectangularity by comparing the area of the detected shape to the contour's area
            float rectangularity = (float)contArea / rectArea;

            // Adjust the threshold as needed
            if (1)
            {
                double area = contourArea(approx);
                if (area > 500)
                {
                    detectedTables.push_back(approx);
                }
            }
        }
    }

    // Display processed image
    for (const auto &table : detectedTables)
    {
        polylines(display, table, true, Scalar(255, 255, 0), 15);
    }
    imshow("Detected Tables", display);
    waitKey(0);
}
