#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

// دالة محاكاة للحصول على شدة الصوت بناءً على رقم الإطار (في الواقع يتم قراءتها من ملف الـ Audio)
float getAudioAmplitude(int frameNumber) {
    // محاكاة موجة صوتية تتغير بين 0 (صمت) و 1 (أعلى صوت) لتبدو الحركة طبيعية مع الكلام
    return std::abs(std::sin(frameNumber * 0.15f)) * 20.0f; 
}

int main() {
    // 1. تحميل كاشف الوجوه ونقاط الملامح من Dlib
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor landmark_predictor;
    
    try {
        // يجب تحميل هذا الملف من موقع dlib ووضعه في نفس مجلد المشروع
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> landmark_predictor;
    } catch (exception& e) {
        cerr << "خطأ: لم يتم العثور على ملف shape_predictor_68_face_landmarks.dat" << endl;
        return -1;
    }

    // 2. فتح فيديو الشخصية (مثلاً: آدم وهو يتحدث في مدينته المظلمة)
    VideoCapture cap("adam_character_video.mp4");
    if (!cap.isOpened()) {
        cerr << "خطأ: لا يمكن فتح ملف الفيديو." << endl;
        return -1;
    }

    // تجهيز نافذة العرض
    namedWindow("Lip Sync - The Weight of Light", WINDOW_AUTOSIZE);
    
    Mat frame;
    int frameCount = 0;

    while (cap.read(frame)) {
        frameCount++;
        
        // تحويل الإطار إلى نظام dlib للتعامل معه
        dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
        
        // كشف الوجوه في الإطار
        std::vector<dlib::rectangle> faces = face_detector(dlib_img);
        
        for (const auto& face : faces) {
            // الحصول على الـ 68 نقطة الخاصة بالوجه
            dlib::full_object_detection shape = landmark_predictor(dlib_img, face);
            
            // نقاط الفم في نظام الـ 68 نقطة تبدأ من النقطة 48 إلى 67
            // الشفة العليا (نقطة 51، 52، 53) والشفة السفلى (نقطة 57، 58, 59)
            
            // جلب شدة الصوت المقابلة للإطار الحالي
            float audio_intensity = getAudioAmplitude(frameCount);
            
            // تعديل النقاط الداخلية للفم برمجياً (تحريك الشفة السفلى للأسفل بناءً على الصوت)
            for (unsigned long i = 55; i <= 59; ++i) {
                // تحريك النقاط عمودياً (Y-axis) لفتح الفم
                Point mouth_point(shape.part(i).x(), shape.part(i).y() + static_cast<int>(audio_intensity));
                
                // رسم تأثير تحريك الفم أو الشفاه باللون الأسود/الوردي لمحاكاة الفتحة
                circle(frame, mouth_point, 2, Scalar(0, 0, 255), -1);
            }
            
            // رسم خطوط الفم الخارجية لتوضيح التحريك (إصدار تجريبي)
            for (unsigned long i = 48; i < 60; ++i) {
                Point p1(shape.part(i).x(), shape.part(i).y() + (i >= 55 ? static_cast<int>(audio_intensity) : 0));
                Point p2(shape.part((i + 1 == 60) ? 48 : i + 1).x(), shape.part((i + 1 == 60) ? 48 : i + 1).y() + ((i + 1 >= 55) ? static_cast<int>(audio_intensity) : 0));
                line(frame, p1, p2, Scalar(0, 255, 0), 2);
            }
        }
        
        // إضافة نصوص الخطافات (Hooks) على الشاشة أثناء العرض لدمج السيناريو
        if (frameCount < 100) {
            putText(frame, "In this world, if you keep a secret... its weight will burn you.", 
                    Point(30, 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        }

        // عرض النتيجة
        imshow("Lip Sync - The Weight of Light", frame);
        
        // إيقاف التشغيل عند الضغط على زر Esc
        if (waitKey(30) == 27) {
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();
    return 0;
}
