#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>

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

        // Check if the point is within both line segments
        if ((min(x1, x2) <= pt.x && pt.x <= max(x1, x2)) && (min(y1, y2) <= pt.y && pt.y <= max(y1, y2)) &&
            (min(x3, x4) <= pt.x && pt.x <= max(x3, x4)) && (min(y3, y4) <= pt.y && pt.y <= max(y3, y4)))
        {
            // Calculate the angle between the two lines
            float m1 = slope(line1);
            float m2 = slope(line2);
            float angle = min(360 - fabs(atan((m2 - m1) / (1 + m1 * m2))), fabs(atan((m2 - m1) / (1 + m1 * m2))));

            // Convert angle to degrees
            angle = angle * 180.0 / CV_PI;

            // Check if the angle is close to 90 degrees within a tolerance
            const float tolerance = 60.0; // Tolerance in degrees
            if (fabs(angle - 90) <= tolerance)
            {
                return pt;
            }
        }
    }
    return Point2f(-1, -1);
}

float areaOfQuadrilateral(const vector<Point2f> &quad)
{
    // Assuming quad has 4 points: A, B, C, D
    float area1 = contourArea(vector<Point2f>{quad[0], quad[1], quad[2]});
    float area2 = contourArea(vector<Point2f>{quad[2], quad[3], quad[0]});
    return area1 + area2; // Total area
}

float rectangularity(const vector<Point2f> &quad)
{
    Rect boundingRectangle = boundingRect(quad); // Changed variable name
    float areaRect = boundingRectangle.width * boundingRectangle.height;
    float areaQuad = areaOfQuadrilateral(quad);
    return areaQuad / areaRect; // Rectangularity
}

void findBestQuadrilateral(const vector<Point2f> &corners, vector<Point2f> &bestQuad, float &maxArea, float rectThreshold)
{
    maxArea = 0;
    for (int i = 0; i < corners.size(); i++)
    {
        for (int j = i + 1; j < corners.size(); j++)
        {
            for (int k = j + 1; k < corners.size(); k++)
            {
                for (int l = k + 1; l < corners.size(); l++)
                {
                    vector<Point2f> quad = {corners[i], corners[j], corners[k], corners[l]};
                    float currArea = areaOfQuadrilateral(quad);
                    float currRect = rectangularity(quad);

                    if (currRect >= rectThreshold && currArea > maxArea)
                    {
                        maxArea = currArea;
                        bestQuad = quad;
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
            // Check if intersection is within the image bounds
            if (pt.x >= 0 && pt.x < src.cols && pt.y >= 0 && pt.y < src.rows)
            {
                corners.push_back(pt);
            }
        }
    }

    // Find the best quadrilateral
    vector<Point2f> bestQuad;
    float maxArea;
    findBestQuadrilateral(corners, bestQuad, maxArea, 0.8); // 0.8 is the rectangularity threshold

    // Draw the best quadrilateral
    if (!bestQuad.empty())
    {
        polylines(combinedImg, bestQuad, true, Scalar(255, 0, 0), 3); // Draw in red
    }

    imshow("Edges and Lines with Corners", combinedImg);
}

void on_trackbar(int, void *)
{
    update_image();
}

int main()
{
    // Hardcoded image path
    string imagePath = "/Users/navi/Desktop/open_cv/Assignment_1/tables/Table1.jpg";

    // Load the image
    src = imread(imagePath, IMREAD_COLOR); // Note: Load in color
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
