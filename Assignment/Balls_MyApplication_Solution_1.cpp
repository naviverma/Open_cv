#include <iostream>
#include <sys/stat.h>
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

        // Apply Gaussian Blur
        // GaussianBlur(src, src, Size(3, 3), 1, 1);

        // imshow("guassian blur",src);
        // waitKey(0);

        // Convert image to float type for kmeans
        Mat data;
        src.convertTo(data, CV_32F);
        data = data.reshape(1, data.total());

        // Use KMeans to cluster the image colors into K clusters
        Mat labels, centers;
        int K = 50;
        kmeans(data, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, 0.2), 4, KMEANS_PP_CENTERS, centers);

        Vec3f white(255, 255, 255);
        Vec3f orange(0, 145, 255);
        vector<int>
            relevantClusters;

        for (int k = 0; k < K; k++)
        {
            Vec3f clusterColor(centers.at<float>(k, 0), centers.at<float>(k, 1), centers.at<float>(k, 2));
            if (colorDistance(clusterColor, white) < 125.0f || colorDistance(clusterColor, orange) < 100.0f)
            {
                relevantClusters.push_back(k);
            }
        }

        Mat clusterImg = src.clone();
        for (int y = 0; y < src.rows; y++)
        {
            for (int x = 0; x < src.cols; x++)
            {
                int label = labels.at<int>(y * src.cols + x);
                if (std::find(relevantClusters.begin(), relevantClusters.end(), label) != relevantClusters.end())
                {
                    Vec3f clusterColor(centers.at<float>(label, 0), centers.at<float>(label, 1), centers.at<float>(label, 2));
                    clusterImg.at<Vec3b>(y, x) = clusterColor;
                }
                else
                {
                    clusterImg.at<Vec3b>(y, x) = Vec3b(0, 0, 0); // color irrelevant clusters with black for clarity
                }
            }
        }

        imshow("Relevant Clusters", clusterImg);
        waitKey(0);

        Mat binImg = Mat::zeros(src.size(), CV_8U);
        for (int y = 0; y < src.rows; y++)
        {
            for (int x = 0; x < src.cols; x++)
            {
                int label = labels.at<int>(y * src.cols + x);
                if (std::find(relevantClusters.begin(), relevantClusters.end(), label) != relevantClusters.end())
                {
                    binImg.at<uchar>(y, x) = 255;
                }
            }
        }

        // Find contours
        vector<vector<Point>> contours;
        findContours(binImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Analyze contours
        vector<pair<Point2f, float>> detectedBalls;
        for (const auto &contour : contours)
        {
            float circularity = (4 * CV_PI * contourArea(contour)) / (pow(arcLength(contour, true), 2));
            if (circularity > 0.4 && circularity <= 1.2)
            {
                Point2f center;
                float radius;
                minEnclosingCircle(contour, center, radius);
                if (radius > 20 && radius < 70) // Size filtering
                {
                    detectedBalls.push_back({center, 2 * radius});
                }
            }
        }

        // Display processed image
        Mat display = src.clone();
        for (const auto &ball : detectedBalls)
        {
            circle(display, ball.first, ball.second / 2, Scalar(0, 255, 0), 2);
            circle(display, ball.first, 2, Scalar(0, 255, 0), -1);
        }

        string outputDirectory = "/Users/navi/Desktop/open_cv/Assignment/balls/Processed_Ball";
        mkdir(outputDirectory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        string outputFilename = outputDirectory + "/Cluster_Image_Solution_1_" + to_string(i + 1) + ".jpg";
        imwrite(outputFilename, clusterImg);

        string outputFilename1 = outputDirectory + "/Final_Image_Solution_1_" + to_string(i + 1) + ".jpg";
        imwrite(outputFilename1, display);

        imshow("Detected Balls", display);
        waitKey(0);

        // Print results
        cout << "Image: " << filenames[i] << endl;
        cout << "Ground Truth vs Detected" << endl;
        for (int j = 0; j < groundTruths[i].size(); j++)
        {
            cout << "Ground Truth: Center=(" << groundTruths[i][j].first.x << ", " << groundTruths[i][j].first.y << "), Diameter=" << groundTruths[i][j].second << endl;
            if (j < detectedBalls.size())
                cout << "Detected: Center=(" << detectedBalls[j].first.x << ", " << detectedBalls[j].first.y << "), Diameter=" << detectedBalls[j].second << endl;
            else
                cout << "Detected: None" << endl;
        }
        cout << "-----------------------------" << endl;
    }

    return 0;
}
