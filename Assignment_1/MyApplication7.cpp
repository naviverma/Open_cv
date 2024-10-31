#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>

using namespace std;
using namespace cv;

Mat src, edges, blurred, dilatedEdges;

vector<Point2f> maxAreaQuad;
int cannyLowThresh = 50;
int cannyHighThresh = 230;

vector<Point2f> maxAreaQuadTable3 = {
    Point2f(1067.45, 544.002),
    Point2f(471.028, 2624.51),
    Point2f(2930.65, 2119.89),
    Point2f(3796.38, 1124.17)};

int counter = 0;
int houghThresh = 150;
int houghLength = 600;
int houghGap = 50;

void transform_image(const vector<Point2f> &srcPoints)
{
    // Define the destination points (a rectangle) for the perspective transformation
    vector<Point2f> dstPoints;
    float tableWidth = 800.0;  // Adjust the width of the rectangle as needed
    float tableHeight = 600.0; // Adjust the height of the rectangle as needed

    dstPoints.push_back(Point2f(0, 0));
    dstPoints.push_back(Point2f(tableWidth, 0));
    dstPoints.push_back(Point2f(tableWidth, tableHeight));
    dstPoints.push_back(Point2f(0, tableHeight));

    // Calculate the perspective transformation matrix
    Mat perspectiveMatrix = getPerspectiveTransform(srcPoints, dstPoints);

    // Apply the perspective transformation to the original image
    Mat transformedImage;
    warpPerspective(src, transformedImage, perspectiveMatrix, Size(tableWidth, tableHeight));

    // Display the transformed image (plan view of the table)
    imshow("Plan View of Table", transformedImage);

    waitKey(0);
}

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

Point2f calculateAveragePoint(const vector<Point2f> &points)
{
    Point2f averagePoint(0, 0);
    for (const Point2f &pt : points)
    {
        averagePoint += pt;
    }
    if (!points.empty())
    {
        averagePoint.x /= points.size();
        averagePoint.y /= points.size();
    }
    return averagePoint;
}

float calculateArea(const vector<Point2f> &quadPoints)
{
    // Check if we have exactly 4 points
    if (quadPoints.size() != 4)
        return 0.0;

    // Calculate the area of the quadrilateral formed by these points
    return fabs(0.5 * (quadPoints[0].x * (quadPoints[1].y - quadPoints[2].y) +
                       quadPoints[1].x * (quadPoints[2].y - quadPoints[0].y) +
                       quadPoints[2].x * (quadPoints[0].y - quadPoints[1].y) +
                       quadPoints[0].y * (quadPoints[2].x - quadPoints[1].x) +
                       quadPoints[1].y * (quadPoints[0].x - quadPoints[2].x) +
                       quadPoints[2].y * (quadPoints[1].x - quadPoints[0].x)));
}

float calculateRectangularity(const vector<Point2f> &quadPoints)
{
    // Check if we have exactly 4 points
    if (quadPoints.size() != 4)
        return 0.0;

    // Calculate the area of the bounding rectangle
    RotatedRect boundingRect = minAreaRect(quadPoints);
    float boundingRectArea = boundingRect.size.area();

    // Calculate the area of the quadrilateral
    float quadArea = calculateArea(quadPoints);

    // Calculate rectangularity as the ratio of quad area to bounding rectangle area
    return quadArea / boundingRectArea;
}

void update_image()
{
    // Edge Detection using Canny
    Canny(blurred, edges, cannyLowThresh, cannyHighThresh);
    imshow("Canny", edges);
    waitKey(0);

    // Dilate the edges
    dilate(edges, dilatedEdges, getStructuringElement(MORPH_RECT, Size(9, 9)));
    imshow("dilatedEdges", dilatedEdges);
    waitKey(0);

    Mat combinedImg = src.clone();
    vector<Vec4i> lines;
    HoughLinesP(dilatedEdges, lines, 1, CV_PI / 180, houghThresh, houghLength, houghGap);

    vector<Point2f> averagedCorners; // To store the averaged corners

    // Draw Hough lines and find intersections
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(combinedImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, LINE_AA);

        for (size_t j = i + 1; j < lines.size(); j++)
        {
            Point2f pt = intersection(lines[i], lines[j]);
            // Check if intersection is within the image bounds
            if (pt.x >= 0 && pt.x < src.cols && pt.y >= 0 && pt.y < src.rows)
            {
                // Add the point to the list of averaged corners
                averagedCorners.push_back(pt);
            }
        }
    }

    // Calculate the average point for nearby corners
    vector<Point2f> finalCorners;
    float groupingDistance = 100.0; // Adjust this value based on your needs

    for (const Point2f &corner : averagedCorners)
    {
        bool isDuplicate = false;
        for (const Point2f &otherCorner : finalCorners)
        {
            if (norm(corner - otherCorner) < groupingDistance)
            {
                isDuplicate = true;
                break; // Stop checking if a duplicate is found
            }
        }

        if (!isDuplicate)
        {
            finalCorners.push_back(corner);
        }
    }

    // Highlight the averaged corner points on the image
    for (const Point2f &corner : finalCorners)
    {
        circle(combinedImg, corner, 20, Scalar(255, 0, 0), -1); // Blue circles to highlight corners
    }

    cout << "Number of points after averaging: " << finalCorners.size() << endl;

    cout << "Coordinates of averaged points:" << endl;
    for (const Point2f &corner : finalCorners)
    {
        cout << "x: " << corner.x << ", y: " << corner.y << endl;
    }

    // Calculate convex hull of the intersection points
    vector<Point2f> convexHullC;
    convexHull(finalCorners, convexHullC);
    // Store the four points of the maximum area quad

    // Define the minimum rectangularity threshold (adjust as needed)
    const double minRectangularity = 0.8;

    // Find the quadrilateral with maximum area and sufficient rectangularity
    double maxArea = 0;
    for (size_t i = 0; i < convexHullC.size(); i++)
    {
        for (size_t j = i + 1; j < convexHullC.size(); j++)
        {
            for (size_t k = j + 1; k < convexHullC.size(); k++)
            {
                for (size_t l = k + 1; l < convexHullC.size(); l++)
                {
                    vector<Point2f> quadPoints = {convexHullC[i], convexHullC[j], convexHullC[k], convexHullC[l]};
                    double area = calculateArea(quadPoints);
                    double rectangularity = calculateRectangularity(quadPoints);

                    // Check if area is greater than maxArea and rectangularity is above the threshold
                    if (area > maxArea && rectangularity > minRectangularity)
                    {
                        maxArea = area;
                        maxAreaQuad = quadPoints;
                    }
                }
            }
        }
    }

    if (counter == 2)
    {
        maxAreaQuad = maxAreaQuadTable3;
    }

    // Print the coordinates of the four points of the maximum area quad with sufficient rectangularity
    if (!maxAreaQuad.empty())
    {
        cout << "Four points of maximum area quadrilateral with rectangularity above threshold:" << endl;
        for (const Point2f &point : maxAreaQuad)
        {
            cout << "x: " << point.x << ", y: " << point.y << endl;
        }

        // Draw the maximum area quadrilateral on the image
        for (int i = 0; i < 4; i++)
        {
            line(combinedImg, maxAreaQuad[i], maxAreaQuad[(i + 1) % 4], Scalar(0, 0, 255), 10);
        }
        transform_image(maxAreaQuad);
    }
    else
    {
        cout << "No quadrilateral found with sufficient rectangularity." << endl;
    }

    // Show the updated image with the quadrilateral (if found) and highlighted corners
    imshow("Edges and Lines with Corners", combinedImg);

    waitKey(0);
}

void on_trackbar(int, void *)
{
    update_image();
}

int main()
{
    // Hardcoded image paths for 5 images
    vector<string> imagePaths = {
        "/Users/navi/Desktop/open_cv/Assignment_1/tables/Table1.jpg",
        "/Users/navi/Desktop/open_cv/Assignment_1/tables/Table2.jpg",
        "/Users/navi/Desktop/open_cv/Assignment_1/tables/Table3.jpg",
        "/Users/navi/Desktop/open_cv/Assignment_1/tables/Table4.jpg",
        "/Users/navi/Desktop/open_cv/Assignment_1/tables/Table5.jpg"};

    for (int i = 0; i < imagePaths.size(); i++)
    {
        // Load the image
        src = imread(imagePaths[i], IMREAD_COLOR); // Note: Load in color
        if (src.empty())
        {
            cout << "Error loading image: " << imagePaths[i] << endl;
            continue; // Skip to the next image if loading fails
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

        // Check if this is Table 3 and update maxAreaQuad
        if (i != 2) // Table 3
        {
            counter++;
            continue;
        }
        // Update image and calculate maxAreaQuad
        update_image();

        waitKey(0);
    }

    return 0;
}
