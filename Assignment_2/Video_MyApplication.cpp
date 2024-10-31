#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <deque>

using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap("/Users/navi/Desktop/open_cv/Assignment_2/TableTennis.avi");
    if (!cap.isOpened())
    {
        cout << "Error: Video file not found." << endl;
        return -1;
    }

    Mat frame;
    Mat prevFrame;
    Mat mask;
    Mat hsv;
    Mat gray;
    Mat ballMask;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    // Parameters for optical flow
    vector<Point2f> prevPts, nextPts;
    vector<uchar> status;
    vector<float> err;
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);

    bool ballOnTable = false;
    bool ballHitByPlayer = false;
    bool ballHitTheNet = false;

    // Parameters for size filtering
    double minBallArea = 100; // Minimum area for a valid ball
    double maxBallArea = 120; // Maximum area for a valid ball

    // Parameters for constraining x-coordinate
    int minX = 125; // Minimum x-coordinate
    int maxX = 775; // Maximum x-coordinate

    // Variables to store previous ball coordinates
    float prevBallX = 0.0;
    float prevBallY = 0.0;

    // Define a data structure to hold past ball positions
    deque<Point2f> pastBallPositions;

    while (cap.read(frame))
    {
        // Convert the frame to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Apply background subtraction
        if (prevFrame.empty())
        {
            gray.copyTo(prevFrame);
        }
        absdiff(prevFrame, gray, mask);
        threshold(mask, mask, 30, 255, THRESH_BINARY);
        morphologyEx(mask, mask, MORPH_CLOSE, element);

        // Convert the frame to the HSV color space
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // Define the color range for the ping pong ball
        Scalar lowerBallColor(29, 86, 6);
        Scalar upperBallColor(64, 255, 255);

        // Threshold the HSV image to get only the ball colors
        inRange(hsv, lowerBallColor, upperBallColor, ballMask);

        // Bitwise-AND mask and original image
        Mat ballOnly;
        bitwise_and(frame, frame, ballOnly, ballMask);

        // Calculate optical flow to track the ball's movement
        if (ballOnTable)
        {
            calcOpticalFlowPyrLK(prevFrame, gray, prevPts, nextPts, status, err, Size(31, 31), 3, termcrit, 0, 0.001);

            // Check if the ball has changed direction (hit by player or hit the table)
            if (status.size() > 0)
            {
                Point2f diff = nextPts[0] - prevPts[0];
                if (norm(diff) > 3.0)
                {
                    if (diff.y > 0)
                    {
                        ballHitByPlayer = true;
                        ballOnTable = false;
                    }
                    else
                    {
                        ballHitTheNet = true;
                        ballOnTable = false;
                    }
                }
            }
        }

        // Detect the ball on the table
        ballMask = mask.clone();
        vector<vector<Point>> contours;
        findContours(ballMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const vector<Point> &contour : contours)
        {
            double area = contourArea(contour);

            if (area >= minBallArea && area <= maxBallArea)
            {
                // Calculate the centroid of the contour
                Moments m = moments(contour);
                Point2f center = Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));

                // Check if the x-coordinate of the centroid is within the specified range
                if (center.x >= minX && center.x <= maxX)
                {
                    ballOnTable = true;
                    prevPts.clear();
                    float radius;
                    minEnclosingCircle(contour, center, radius);
                    circle(frame, center, (int)radius, Scalar(0, 0, 255), 2);

                    // Store the current ball coordinates
                    prevBallX = center.x;
                    prevBallY = center.y;
                    prevPts.push_back(center);
                }
            }
        }

        // Add the current position to the deque
        if (ballOnTable)
        {
            pastBallPositions.push_back(Point2f(prevBallX, prevBallY));

            // If the deque size exceeds N, remove the oldest entry
            int N = 10; // Adjust N to your desired number of frames to consider
            if (pastBallPositions.size() > N)
            {
                pastBallPositions.pop_front();
            }

            // Calculate the average position from the past frames
            Point2f smoothedPosition(0, 0);
            for (const Point2f &pos : pastBallPositions)
            {
                smoothedPosition += pos;
            }
            if (!pastBallPositions.empty())
            {
                smoothedPosition.x /= static_cast<float>(pastBallPositions.size());
                smoothedPosition.y /= static_cast<float>(pastBallPositions.size());
            }

            // Use `smoothedPosition` instead of `prevBallX` and `prevBallY` for further processing
            prevBallX = smoothedPosition.x;
            prevBallY = smoothedPosition.y;
        }

        // Show the processed frame
        imshow("Table Tennis Video", frame);

        // Check if the ball changed direction and label it accordingly
        if (ballOnTable)
        {
            if (ballHitByPlayer)
            {
                cout << "Frame: " << cap.get(CAP_PROP_POS_FRAMES) << " - Ball hit by player" << endl;
                cout << "Ball Coordinates: (" << prevBallX << ", " << prevBallY << ")" << endl;
                ballHitByPlayer = false;
            }
            else if (ballHitTheNet)
            {
                cout << "Frame: " << cap.get(CAP_PROP_POS_FRAMES) << " - Ball hit the table" << endl;
                cout << "Ball Coordinates: (" << prevBallX << ", " << prevBallY << ")" << endl;
                ballHitTheNet = false;
            }
        }

        if (waitKey(30) == 27)
        {
            break;
        }

        // Store the current frame as the previous frame for the next iteration
        gray.copyTo(prevFrame);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
