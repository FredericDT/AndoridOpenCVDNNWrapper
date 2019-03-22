#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_onl_fdt_java_android_odappliedwsc_JavaActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
