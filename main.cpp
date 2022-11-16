#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <cstring>
#include <opencv2/opencv.hpp>
#include "timing.h"

using namespace cv;
using namespace std;
#ifndef MIN
#define MIN(a, b)    ( (a) > (b) ? (b) : (a) )
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#define CLIP3(x,min,max)         ( (x)< (min) ? (min) : ((x)>(max)?(max):(x)) )

#define PI acos(-1)

//#define int8    char
#define uint8   unsigned char
#define int16   short
#define uint16  unsigned short
#define int32   int
#define uint32  unsigned int
#define int64   long long
#define uint64  unsigned long long


//Algorithm 1
/*
 * in_image / out_image - input / output image
 * img_width / img_height - input image width / height
 * r - gaussian blur kernel sigma
 */
uint8 gaussian_blur_1(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r,uint8 radius){
    //r - gauss sigma
    //uint8 rs = ceil(2.57*r); //gass kernel size
    uint8 rs = radius; //gass kernel size
    //printf("rs:%d\n",rs);
    for(int h = 0;h < img_height;h++){
        for(int v = 0;v<img_width;v++){
            float val_wgt =0;
            float sum_wgt =0;

            for(int i = h-rs;i < h+rs+1;i++){
                for(int j = v-rs;j<v+rs+1;j++){
                uint16 y = MIN(img_height -1,MAX(0,i));
                uint16 x = MIN(img_width-1,MAX(0,j));
                int dist = (y-i)*(y-i) + (x-j)*(x-j);

                float wgt = exp(-dist/(2*r*r))/(PI*2*r*r);
                val_wgt += in_image[y*img_width+x]*wgt;
                sum_wgt += wgt;
                }
            }
            out_image[h*img_width+v] = CLIP3(round(val_wgt/sum_wgt),0,255);
        }
    }

    return 0;
}

//Algorithm 2 box blur
void boxesForGauss(uint8 sigma,uint8 n ,uint8* sizes){ //gauss sigma / numbers of boxs

    float wIdeal = sqrt((12.0*sigma*sigma/n)+1.0);  // Ideal averaging filter width
    int wl = floor(wIdeal);
    if(wl%2==0)
        wl--;
    int wu = wl+2;

    float mIdeal = (12.0*sigma*sigma - n*wl*wl - 4*n*wl - 3*n)/(-4*wl - 4);
    int m = round(mIdeal);
    for(int i=0; i<n; i++){
        sizes[i] = i<m?wl:wu;
        printf("sizes:%d,%d\n",i,sizes[i]);
    }

}
void boxBlur_2(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){

    for(int h = 0;h < img_height;h++){
        for(int v = 0;v<img_width;v++){
            float val_sum =0;
            for(int i = h-r;i < h+r+1;i++){
                for(int j = v-r;j<v+r+1;j++){
                    uint16 y = MIN(img_height -1,MAX(0,i));
                    uint16 x = MIN(img_width-1,MAX(0,j));

                    val_sum += in_image[y*img_width+x];
                }
            }
            float temp_val = 1.0*val_sum/((r+r+1)*(r+r+1));

            out_image[h*img_width+v] = CLIP3(round(temp_val),0,255);
        }
    }
}
void gaussBlur_2(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){

    uint8 bxs[3] = {0};
    boxesForGauss(r,3,bxs);
    //printf("%d,%d,%d\n",bxs[0],bxs[1],bxs[2]);
    boxBlur_2(in_image,out_image,img_width,img_height,(bxs[0]-1)/2);
    boxBlur_2(out_image,in_image,img_width,img_height,(bxs[1]-1)/2);
    boxBlur_2(in_image,out_image,img_width,img_height,(bxs[2]-1)/2);
 }

 // Algorithm 3
 void gaussBlur_3(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r,uint8 radius){
    //r - gauss sigma
     //uint8 rs = ceil(2.57*r); //gass kernel size
     uint8 rs = radius; //gass kernel size
     float deno = 1.0 / (r * sqrt(2.0 * PI));
     float nume = -1.0 / (2.0 * r * r);
     float* gauss_kernel = (float*) malloc(sizeof (float)*(r+r+1));
     float sum_kernel = 0.0;
     //高斯分布产生的数组
     for (int i = 0, x = -rs; x < rs+1; ++x, ++i) {
         float g = deno * exp(1.0 * nume * x * x);

         gauss_kernel[i] = g;
         sum_kernel += g;
     }
     //gauss_kernel 归一化
     int len = rs + rs + 1;
     for (int i = 0; i < len; ++i)
         gauss_kernel[i] /= sum_kernel;

     uint8* rowData = (uint8*) malloc(img_width * sizeof(uint8));
     uint8* colData = (uint8*) malloc(img_height * sizeof(uint8));

     for(int h=0;h<img_height;h++){
         memcpy(rowData, in_image + h * img_width, sizeof(uint8) * img_width);
         for(int v=0;v<img_width;v++){
             float sum_val = 0.0;
             float sum_gauss =0.0;
             for(int i=-rs;i<rs+1;i++){
                 uint16 k = MIN(img_width -1,MAX(0,i+v));
                 uint8 pixel_val = rowData[k];
                 sum_val +=pixel_val*gauss_kernel[i+rs];
                 sum_gauss +=gauss_kernel[i+rs];
             }
             in_image[h*img_width+v] = CLIP3(round(sum_val/sum_gauss),0,255);
         }
     }

     for(int vv = 0;vv < img_width;vv++){
         for (int y = 0; y < img_height; ++y)
             colData[y] = in_image[y * img_width + vv];
         for(int hh = 0;hh < img_height;hh++){
             //colData[hh] = in_image[hh*img_width + vv];
             float sum_val = 0.0;
             float sum_gauss =0.0;
             for(int i = -rs;i< rs+1;i++){
                 uint16 k = MIN(img_height -1,MAX(0,i+hh));
                 uint8 pixel_val = colData[k];
                 sum_val +=pixel_val*gauss_kernel[i+rs];
                 sum_gauss +=gauss_kernel[i+rs];
             }
             out_image[hh*img_width+vv] = CLIP3(round(sum_val/sum_gauss),0,255);
         }
     }
     free(colData);
     free(rowData);
 }

 // Algorithm 4
 void boxBlurH_4(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){
     for(int h = 0;h < img_height;h++){
         for(int v = 0;v<img_width;v++){
             float val_sum =0;
             for(int j = v-r;j<v+r+1;j++){
                 uint16 x = MIN(img_width-1,MAX(0,j));
                 val_sum += in_image[h*img_width + x];
             }
             out_image[h*img_width + v] = CLIP3(round(val_sum/(r+r+1)),0,255);
         }
     }
}
void boxBlurT_4(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){
    for(int h = 0;h < img_height;h++){
        for(int v = 0;v<img_width;v++){
            float val_sum =0;
            for(int j = h-r;j<h+r+1;j++){
                uint16 y = MIN(img_height-1,MAX(0,j));
                val_sum += in_image[y*img_width + v];
            }
            out_image[h*img_width + v] = CLIP3(round(val_sum/(r+r+1)),0,255);
        }
    }
}
void boxBlur_4(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){
    boxBlurH_4(in_image,in_image,img_width,img_height,r);
    boxBlurT_4(in_image,out_image,img_width,img_height,r);
 }
void gaussBlur_4(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){

    uint8 bxs[3] = {0};
    boxesForGauss(r,3,bxs);
    //printf("%d,%d,%d\n",bxs[0],bxs[1],bxs[2]);
    boxBlur_4(in_image,out_image,img_width,img_height,(bxs[0]-1)/2);
    boxBlur_4(out_image,in_image,img_width,img_height,(bxs[1]-1)/2);
    boxBlur_4(in_image,out_image,img_width,img_height,(bxs[2]-1)/2);
}

// Algorithm 5
#if 1
 void boxBlurH_5(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){

    float  iarr = 1.0/(r+r+1);// radius range on either side of a pixel + the pixel itself

    for(int i = 0;i < img_height;i++){
        int ti = i*img_width;//pixel index; will traverse the width of the image for each loop
        int li = ti;// trailing pixel index
        int ri = ti + r;//pixel index of the furthest reach of the radius

        int fv = in_image[ti];// first pixel value of the row
        int lv = in_image[ti+img_width-1];// last pixel value in the row

        // create a "value accumulator" - we will be calculating the average of pixels
        // surrounding each one - is faster to add newest value, remove oldest, and
        // then average. This initial value is for pixels outside image bounds
        int val = (r+1)*fv;
        // for length of radius, accumulate the total value of all pixels from current pixel
        // index and record it into the target channel first pixel
        for(int j=0; j< r; j++)
            val += in_image[ti+j];

        // for the next $boxRadius pixels in the row, record pixel value of average of all
        // pixels within the radius and save average into target channel
        for(int j=0; j<= r ; j++) {
            val += in_image[ri++] - fv;
            out_image[ti++] = round(val*iarr);
        }

        // now that we've completely removed the overflow pixels from the value accumulator,
        // continue on, adding new values, removing old ones, and averaging the
        // accumulated value
        for(int j=r+1; j<img_width-r; j++) {
            val += in_image[ri++] - in_image[li++];
            out_image[ti++] = round(val*iarr);
        }
        // finish off the row of pixels, duplicating the edge pixel instead of going out of image bounds
        for(int j=img_width-r; j<img_width ; j++) {
            val += lv - in_image[li++];
            out_image[ti++] = round(val*iarr);
        }
    }
}
void boxBlurT_5(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){
    float  iarr = 1.0/(r+r+1);
    for(int i = 0;i < img_width;i++){
        int ti = i;
        int li = ti;
        int ri = ti + r*img_width;
        int fv = in_image[ti];
        int lv = in_image[ti + img_width*(img_height- 1)];
        int val = (r+1)*fv;
        for(int j=0; j<r; j++)
            val += in_image[ti+j*img_width];
        for(int j=0; j<=r ; j++) {
            val += in_image[ri] - fv;
            out_image[ti] = round(val*iarr);
            ri +=img_width;
            ti +=img_width;
        }
        for(int j=r+1; j<img_height-r; j++) {
            val += in_image[ri] - in_image[li];
            out_image[ti] = round(val*iarr);
            li +=img_width;
            ri +=img_width;
            ti +=img_width;
        }
        for(int j=img_height-r; j<img_height ; j++) {
            val += lv - in_image[li];
            out_image[ti] = round(val*iarr);
            li +=img_width;
            ti +=img_width;
        }
    }
}

#endif
#if 0
void boxBlurH_5(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){

    for(int i = 0;i < img_height;i++){
        int accumulation = r * in_image[i*img_width];
        for(int j=0;j <r;j++){
            accumulation += in_image[i*img_width + j];
        }
        out_image[i*img_width] = round(accumulation/(2.0*r + 1));

        for (int j = 1; j < img_width; j++) {
            int left = MAX(0, j - r - 1);
            int right = MIN(img_width - 1, j + r);
            accumulation =accumulation + (in_image[i * img_width + right] - in_image[i * img_width + left]);
            out_image[i * img_width + j] = round(accumulation / (2.0 * r + 1));
        }

    }
}
void boxBlurT_5(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){
    for (int i = 0; i < img_width; i++) {
        int accumulation = r * in_image[i];
        for (int j = 0; j <= r; j++) {
            accumulation += in_image[j * img_width + i];
        }
        out_image[i] = round(accumulation / (2 * r + 1));
        for (int j = 1; j < img_height; j++) {
            int top = MAX(0, j - r - 1);
            int bottom = MIN(img_height - 1, j + r);
            accumulation =accumulation + in_image[bottom * img_width + i] - in_image[top * img_width + i];
            out_image[j * img_width + i] = round(accumulation / (2.0 * r + 1));
        }
    }
}
#endif
void boxBlur_5(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){
    uint8* temp_buffer = (uint8*) malloc(img_width * img_height * sizeof(uint8));
    if(temp_buffer){
        memset(temp_buffer, 0, img_width * img_height*sizeof(uint8));
    }
    boxBlurH_5(in_image,temp_buffer,img_width,img_height,r);
    boxBlurT_5(temp_buffer,out_image,img_width,img_height,r);
    free(temp_buffer);
}
void gaussBlur_5(uint8* in_image,uint8* out_image,int img_width,int img_height,uint8 r){

    uint8 bxs[3] = {0};
    boxesForGauss(r,3,bxs);
    //printf("%d,%d,%d\n",bxs[0],bxs[1],bxs[2]);
    boxBlur_5(in_image,out_image,img_width,img_height,(bxs[0]-1)/2);
    boxBlur_5(out_image,in_image,img_width,img_height,(bxs[1]-1)/2);
    boxBlur_5(in_image,out_image,img_width,img_height,(bxs[2]-1)/2);
}

uint8 split_image(uint8* in_image,uint8* r_image,uint8* g_image,uint8* b_image,int img_width,int img_height){

    for(int h = 0;h<img_height;h++){
        for(int v = 0;v<img_width;v++){
            r_image[h*img_width+v] = in_image[3*h*img_width+3*v];
            g_image[h*img_width+v] = in_image[3*h*img_width+3*v+1];
            b_image[h*img_width+v] = in_image[3*h*img_width+3*v+2];
        }
    }
    return 0;
}

uint8 merge_image(uint8* out_image,uint8* r_image,uint8* g_image,uint8* b_image,int img_width,int img_height){

    for(int h = 0;h<img_height;h++){
        for(int v = 0;v<img_width;v++){
            out_image[3*(h*img_width+v)] = r_image[h*img_width+v];
            out_image[3*(h*img_width+v)+1] = g_image[h*img_width+v];
            out_image[3*(h*img_width+v)+2] = b_image[h*img_width+v];
        }
    }
    return 0;
}
int main() {
    //cout << "Hello, World!" << endl;
    char* cur_file = "test.bmp";
    int Width = 0;
    int Height = 0;

    Mat cur_img;
    cur_img = imread(cur_file);
    Width = cur_img.cols;
    Height = cur_img.rows;
    //imshow("cur_IMG",cur_img);
    //waitKey();
    uint8* input_img = (uint8*) malloc(Width * Height * sizeof(uint8)*3);
    if(input_img){
        memcpy(input_img, cur_img.data, Width * Height*sizeof(uint8)*3);
    }
    uint8* output_img = (uint8*) malloc(Width * Height * sizeof(uint8)*3);
    if(output_img){
        memset(output_img, 0, Width * Height*sizeof(uint8)*3);
    }
    uint8* r_img = (uint8*) malloc(Width * Height * sizeof(uint8));
    if(r_img){
        memset(r_img, 0, Width * Height*sizeof(uint8));
    }
    uint8* r_out = (uint8*) malloc(Width * Height * sizeof(uint8));
    if(r_out){
        memset(r_out, 0, Width * Height*sizeof(uint8));
    }
    uint8* g_img = (uint8*) malloc(Width * Height * sizeof(uint8));
    if(g_img){
        memset(g_img, 0, Width * Height*sizeof(uint8));
    }
    uint8* g_out = (uint8*) malloc(Width * Height * sizeof(uint8));
    if(g_out){
        memset(g_out, 0, Width * Height*sizeof(uint8));
    }
    uint8* b_img = (uint8*) malloc(Width * Height * sizeof(uint8));
    if(b_img){
        memset(b_img, 0, Width * Height*sizeof(uint8));
    }
    uint8* b_out = (uint8*) malloc(Width * Height * sizeof(uint8));
    if(b_out){
        memset(b_out, 0, Width * Height*sizeof(uint8));
    }

    //Algorithm 1 //450ms
#if 0
    split_image(input_img,r_img,g_img,b_img,Width,Height);
    double startTime = now();
    for(int i=0;i<10;i++){
        gaussian_blur_1(r_img,r_img,Width,Height,1,1);
    }
    double nLoadTime = calcElapsed(startTime, now());
    printf("process time: %d ms.\n ", (int) (nLoadTime * 1000/10));
    gaussian_blur_1(g_img,g_img,Width,Height,1,1);
    gaussian_blur_1(b_img,b_img,Width,Height,1,1);

    merge_image(output_img,r_img,g_img,b_img,Width,Height);
    Mat dst_image = Mat(1080,1920,CV_8UC3,(uint8*)output_img);
    imshow("dst_image",dst_image);
    waitKey();
    char* output_image_name = "dst_img_1.bmp";
    imwrite(output_image_name,dst_image);
#endif

    //Algorithm 2 //222ms
#if 0
    split_image(input_img,r_img,g_img,b_img,Width,Height);
    double startTime = now();
    for(int i=0;i<10;i++) {
        gaussBlur_2(r_img, r_out, Width, Height, 1);
    }
    double nLoadTime = calcElapsed(startTime, now());
    printf("process time: %d ms.\n ", (int) (nLoadTime * 1000/10));
    gaussBlur_2(g_img,g_out,Width,Height,1);
    gaussBlur_2(b_img,b_out,Width,Height,1);

    merge_image(output_img,r_out,g_out,b_out,Width,Height);
    Mat dst_image = Mat(1080,1920,CV_8UC3,(uint8*)output_img);
    imshow("dst_image",dst_image);
    waitKey();
    char* output_image_name = "dst_img_2.bmp";
    imwrite(output_image_name,dst_image);
#endif

    //Algorithm 3 //159ms
#if 0
    split_image(input_img,r_img,g_img,b_img,Width,Height);
    double startTime = now();
    for(int i=0;i<10;i++) {
        gaussBlur_3(r_img, r_img, Width, Height, 1, 1);
    }
    double nLoadTime = calcElapsed(startTime, now());
    printf("process time: %d ms.\n ", (int) (nLoadTime * 1000/10));
    gaussBlur_3(g_img,g_img,Width,Height,1,1);
    gaussBlur_3(b_img,b_img,Width,Height,1,1);

    merge_image(output_img,r_img,g_img,b_img,Width,Height);
    Mat dst_image = Mat(1080,1920,CV_8UC3,(uint8*)output_img);
    imshow("dst_image",dst_image);
    waitKey();
    char* output_image_name = "dst_img_3.bmp";
    imwrite(output_image_name,dst_image);

#endif
    //Algorithm 4 //386ms
#if 0
    split_image(input_img,r_img,g_img,b_img,Width,Height);
    double startTime = now();
    for(int i=0;i<10;i++) {
        gaussBlur_4(r_img, r_img, Width, Height, 1);
    }
    double nLoadTime = calcElapsed(startTime, now());
    printf("process time: %d ms.\n ", (int) (nLoadTime * 1000/10));
    gaussBlur_4(g_img,g_img,Width,Height,1);
    gaussBlur_4(b_img,b_img,Width,Height,1);

    merge_image(output_img,r_img,g_img,b_img,Width,Height);
    Mat dst_image = Mat(1080,1920,CV_8UC3,(uint8*)output_img);
    imshow("dst_image",dst_image);
    waitKey();
    char* output_image_name = "dst_img_4.bmp";
    imwrite(output_image_name,dst_image);
#endif
    //Algorithm 5 //150ms
#if 0
    split_image(input_img,r_img,g_img,b_img,Width,Height);
    double startTime = now();
    for(int i=0;i<10;i++){
        gaussBlur_5(r_img,r_out,Width,Height,1);
    }
    double nLoadTime = calcElapsed(startTime, now());
    printf("process time: %d ms.\n ", (int) (nLoadTime * 1000/10));
    gaussBlur_5(g_img,g_out,Width,Height,1);
    gaussBlur_5(b_img,b_out,Width,Height,1);

    merge_image(output_img,r_out,g_out,b_out,Width,Height);
    Mat dst_image = Mat(1080,1920,CV_8UC3,(uint8*)output_img);
    imshow("dst_image",dst_image);
    waitKey();
    char* output_image_name = "dst_img_5.bmp";
    imwrite(output_image_name,dst_image);
#endif

    free(input_img);
    free(output_img);
    free(r_img);
    free(r_out);
    free(g_img);
    free(g_out);
    free(b_img);
    free(b_out);

    return 0;
}
