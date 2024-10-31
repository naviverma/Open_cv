#include <iostream>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
    vector<string> filenames = {
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball1.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball2.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball3.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball4.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball5.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball6.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball7.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball8.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball9.jpg",
        "/Users/navi/Desktop/open_cv/Assignment/balls/Ball10.jpg"};

    vector<vector<pair<Point2f, float>>> groundTruths = {
        {{Point2f(564.5, 311.5), 83}},
        {{Point2f(432, 456.5), 136.5}},
        {{Point2f(414.5, 407), 95.5}},
        {{Point2f(363, 380), 113}},
        {{Point2f(146.5, 472), 85.5},
         {Point2f(440.5, 362), 84.5},
         {Point2f(711.5, 481.5), 84}},
        {{Point2f(383.5, 330.5), 96}},
        {{Point2f(529, 282), 70}},
        {{Point2f(523.5, 458.5), 61}},
        {{Point2f(531.5, 403.5), 59}},
        {{Point2f(494.5, 215.5), 41}}};

    for (int i = 0; i < filenames.size(); i++)
    {
        Mat src = imread(filenames[i], IMREAD_COLOR);

        // Convert to grayscale
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        imshow("gray", gray);
        waitKey(0);

        // Apply Gaussian blur to reduce noise and improve edge detection
        GaussianBlur(gray, gray, Size(9, 9), 2, 2);

        imshow("Processed Frame", gray);
        waitKey(0);

        // Use the Hough transform to detect circles
        vector<Vec3f> circles;
        HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 200, 40, 20, 70);

        // Display the results
        Mat display = src.clone();
        for (size_t j = 0; j < circles.size(); j++)
        {
            Point center(cvRound(circles[j][0]), cvRound(circles[j][1]));
            int radius = cvRound(circles[j][2]);
            circle(display, center, 3, Scalar(0, 255, 0), -1, 8, 0);     // circle center
            circle(display, center, radius, Scalar(0, 255, 0), 2, 8, 0); // circle outline
        }
        string outputDirectory = "/Users/navi/Desktop/open_cv/Assignment/balls/Processed_Ball";
        mkdir(outputDirectory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        string outputFilename = outputDirectory + "/Final_Image_Solution_2_" + to_string(i + 1) + ".jpg";
        imwrite(outputFilename, display);
        imshow("Detected Balls", display);
        waitKey(0);

        // Print results
        cout << "Image: " << filenames[i] << endl;
        cout << "Ground Truth vs Detected" << endl;
        for (int j = 0; j < groundTruths[i].size(); j++)
        {
            cout << "Ground Truth: Center=(" << groundTruths[i][j].first.x << ", " << groundTruths[i][j].first.y << "), Diameter=" << groundTruths[i][j].second << endl;
            if (j < circles.size())
                cout << "Detected: Center=(" << circles[j][0] << ", " << circles[j][1] << "), Diameter=" << 2 * circles[j][2] << endl;
            else
                cout << "Detected: None" << endl;
        }
        cout << "-----------------------------" << endl;
    }

    return 0;
}
