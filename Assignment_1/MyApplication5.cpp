#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>
#include <vector>

using namespace std;
using namespace cv;

Mat src, edges, blurred, dilatedEdges;

int cannyLowThresh = 50;
int cannyHighThresh = 230;

int houghThresh = 150;
int houghLength = 600;
int houghGap = 50;

float slope(Vec4i line)
{
    // Handle vertical lines
    if (line[2] == line[0])
    {
        return std::numeric_limits<float>::max(); // A very large value to represent infinity
    }
    return (float)(line[3] - line[1]) / (line[2] - line[0]);
}

Point2f intersection(Vec4i line1, Vec4i line2)
{
    Point2f pt;
    int x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
    int x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];

    float a1 = y2 - y1;
    float b1 = x1 - x2;
    float c1 = a1 * (x1) + b1 * (y1);

    float a2 = y4 - y3;
    float b2 = x3 - x4;
    float c2 = a2 * (x3) + b2 * (y3);

    float determinant = a1 * b2 - a2 * b1;

    if (determinant != 0)
    {
        pt.x = (b2 * c1 - b1 * c2) / determinant;
        pt.y = (a1 * c2 - a2 * c1) / determinant;
        return pt;
    }
    return Point2f(-1, -1);
}

void findLargestQuadrilateral(const vector<Point2f> &corners, vector<Point2f> &largestQuad, float &maxArea)
{
    const float threshold = 0.8; // Set a threshold for rectangularity (e.g., 0.8)

    for (size_t i = 0; i < corners.size(); ++i)
    {
        for (size_t j = i + 1; j < corners.size(); ++j)
        {
            for (size_t k = j + 1; k < corners.size(); ++k)
            {
                for (size_t l = k + 1; l < corners.size(); ++l)
                {
                    vector<Point2f> quad = {corners[i], corners[j], corners[k], corners[l]};
                    float quadArea = contourArea(quad);
                    Rect boundingBox = boundingRect(quad);
                    float boxArea = boundingBox.width * boundingBox.height;
                    float rectangularity = quadArea / boxArea;

                    if (rectangularity >= threshold && quadArea > maxArea)
                    {
                        maxArea = quadArea;
                        largestQuad = quad;
                    }
                }
            }
        }
    }
}

void update_image()
{
    // Edge Detection using Canny
    Canny(blurred, edges, cannyLowThresh, cannyHighThresh);

    // Dilate the edges
    dilate(edges, dilatedEdges, getStructuringElement(MORPH_RECT, Size(9, 9)));

    Mat combinedImg = src.clone();
    vector<Vec4i> lines;
    HoughLinesP(dilatedEdges, lines, 1, CV_PI / 180, houghThresh, houghLength, houghGap);

    vector<Point2f> corners;

    // Draw Hough lines and find intersections
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(combinedImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 5, LINE_AA);

        for (size_t j = i + 1; j < lines.size(); j++)
        {
            Point2f pt = intersection(lines[i], lines[j]);
            if (pt.x >= 0 && pt.x < src.cols && pt.y >= 0 && pt.y < src.rows)
            {
                corners.push_back(pt);
            }
        }
    }

    // Find the largest quadrilateral with sufficient rectangularity
    vector<Point2f> largestQuad;
    float maxArea = 0;
    findLargestQuadrilateral(corners, largestQuad, maxArea);

    // Draw the largest quadrilateral
    for (int i = 0; i < largestQuad.size(); i++)
    {
        line(combinedImg, largestQuad[i], largestQuad[(i + 1) % largestQuad.size()], Scalar(255, 0, 0), 2);
    }

    imshow("Edges and Lines with Corners", combinedImg);
}

void on_trackbar(int, void *)
{
    update_image();
}

int main()
{
    // Load the image
    src = imread("/Users/navi/Desktop/open_cv/Assignment_1/tables/Table5.jpg", IMREAD_COLOR); // Replace with your image path
    if (src.empty())
    {
        cout << "Error loading image" << endl;
        return -1;
    }

    cvtColor(src, blurred, COLOR_BGR2GRAY);
    GaussianBlur(blurred, blurred, Size(5, 5), 3, 3);

    // Create windows and trackbars
    namedWindow("Edges and Lines with Corners", WINDOW_NORMAL);
    createTrackbar("Canny Low Threshold", "Edges and Lines with Corners", &cannyLowThresh, 500, on_trackbar);
    createTrackbar("Canny High Threshold", "Edges and Lines with Corners", &cannyHighThresh, 500, on_trackbar);
    createTrackbar("Hough Threshold", "Edges and Lines with Corners", &houghThresh, 500, on_trackbar);
    createTrackbar("Min Line Length", "Edges and Lines with Corners", &houghLength, 200, on_trackbar);
    createTrackbar("Max Line Gap", "Edges and Lines with Corners", &houghGap, 100, on_trackbar);

    update_image(); // Initial display

    waitKey(0);
    return 0;
}
