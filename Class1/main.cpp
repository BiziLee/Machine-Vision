#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

int main()
{
    // 图片放在项目根目录。使用绝对路径，确保从 Visual Studio 启动时也能找到图片。
    const std::string imagePath = "F:/Things/OpenCV/C++Project/Class1/see.png";

    // 以灰度模式读取图片；大津法使用单通道灰度图进行阈值分割。
    const cv::Mat gray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (gray.empty())
    {
        std::cerr << "无法读取图片：" << imagePath << std::endl;
        std::cerr << "请确认图片路径正确，且 see.png 文件存在。" << std::endl;
        return 1;
    }

    // 高斯模糊可减弱传感器噪声和纸面细微纹理，减少阈值分割产生的小碎片。
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 回形针比背景暗，因此使用 THRESH_BINARY_INV：低于阈值的暗像素变为白色前景。
    // THRESH_OTSU 会自动计算大津阈值，阈值参数 0 会由 OpenCV 自动替代。
    cv::Mat binary;
    const double otsuThreshold = cv::threshold(
        blurred,
        binary,
        0,
        255,
        cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 回形针边缘有少量细小断口；闭运算（先膨胀再腐蚀）连接相邻的前景边缘。
    // 这里使用椭圆形 5x5 核，尽量平滑地连接断口而不明显改变外形。
    const cv::Mat morphologyKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(5, 5));
    cv::Mat connectedBinary;
    cv::morphologyEx(
        binary,
        connectedBinary,
        cv::MORPH_CLOSE,
        morphologyKernel);

    // connectedBinary 中白色像素是前景。连通域分析会为每个彼此连接的白色区域编号。
    // stats 每行依次包含区域的左上角、宽高和像素面积；centroids 保存区域质心。
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int labelCount = cv::connectedComponentsWithStats(
        connectedBinary,
        labels,
        stats,
        centroids,
        8,          // 8 邻域：斜向接触的像素也视为连通。
        CV_32S);

    // 标签 0 是图像背景，不是回形针；下面再按面积排除小噪点。
    // 该面积下限适用于当前图片尺寸，可根据其他图片的分辨率调整。
    const int minimumArea = 500;

    // 输出图像使用彩色原图，方便观察原物体以及检测框和编号。
    cv::Mat annotated;
    cv::cvtColor(gray, annotated, cv::COLOR_GRAY2BGR);

    int paperclipCount = 0;
    for (int label = 1; label < labelCount; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);

        // 小于面积下限的连通区域通常是纸面纹理或阈值产生的噪点，不计为物体。
        if (area < minimumArea)
        {
            continue;
        }

        ++paperclipCount;

        // 读取当前连通域的外接矩形，并在图上画绿色边框。
        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const cv::Rect bounds(left, top, width, height);
        cv::rectangle(annotated, bounds, cv::Scalar(0, 220, 0), 2);

        // 在外接矩形上方标注序号；若物体靠近图片顶部，就把文字放到框内。
        const cv::Point textPosition(
            left,
            (top >= 10) ? top - 6 : top + 22);
        cv::putText(
            annotated,
            std::to_string(paperclipCount),
            textPosition,
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 0, 255),
            2,
            cv::LINE_AA);

        // 同时在控制台打印编号、面积和质心，便于核对每个连通域。
        std::cout << "回形针 " << paperclipCount
                  << "：面积 = " << area << " 像素，质心 = ("
                  << centroids.at<double>(label, 0) << ", "
                  << centroids.at<double>(label, 1) << ")" << std::endl;
    }

    // 将总数写在图像左上角；使用深色底框以保证文字在浅色背景上清晰可见。
    const std::string countText = "Count: " + std::to_string(paperclipCount);
    cv::rectangle(annotated, cv::Point(10, 10), cv::Point(250, 52), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(
        annotated,
        countText,
        cv::Point(20, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        0.9,
        cv::Scalar(0, 0, 200),
        2,
        cv::LINE_AA);

    // 控制台显示大津法计算出的阈值和最终计数。
    std::cout << "大津法阈值：" << otsuThreshold << std::endl;
    std::cout << "回形针总数：" << paperclipCount << std::endl;

    // 显示灰度图、二值图和标注结果，便于检查分割与连通域效果。
    cv::imshow("Grayscale Image", gray);
    cv::imshow("Otsu Binary Image", binary);
    cv::imshow("Connected Components - Paperclip Count", annotated);
    cv::waitKey(0);

    return 0;
}
