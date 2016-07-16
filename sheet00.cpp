#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <fstream>
#include <cstdio>
#include <climits>

#include <iterator>
#include <vector>
#include <stdint.h>
#include <iomanip>
#include <math.h>
#include <complex>
#include <cmath>
#include <algorithm>
#include <vector>



#define LIGHT 299792458.0

using namespace std;
using namespace cv;

typedef complex<double> dcomp;
typedef std::vector<float> row_type;
typedef std::vector<row_type> matrix_type;

int rows_ir=424;
int cols_ir=512;


template <typename T>
long atoi(const std::basic_string<T> &str){
	std::basic_stringstream<T> stream(str);
	long res;
	return !(stream >>res)?0:res;
}

float  interpolate(float value, float leftMin, float leftMax, float rightMin, float rightMax){
    float leftSpan = leftMax - leftMin;
    float rightSpan = rightMax - rightMin;

    float valueScaled = (value - leftMin) / (leftSpan);

    float value_to_return= rightMin + (valueScaled * rightSpan);
    // std::cout << "value to return" << value_to_return << std::endl;
    return value_to_return;
}


matrix_type  intersect_saw  (int freq, int num_wraps, float val, int num_points, float sigma){

  float c=299792458.0;

  matrix_type intersects;
  intersects.resize(num_wraps);
  for (size_t i = 0; i < num_wraps; i++) {
    intersects[i].resize(num_points);
  }

  std::vector<float> points(num_points);
  int middle_point=int (floor (num_points/2));
  float  first_dist= c * (2.0*M_PI) /(4.0*M_PI*freq*1000000.0) ;

  //  std::cout << "freq " << freq << " has mid point " << middle_point << " fist dist " << first_dist << std::endl;

  //create the vector of points
  points[middle_point]=val;
  for (size_t i = middle_point+1; i < num_points; i++) {
    // std::cout << "high value, " << i << std::endl;
    int dist_middle=abs(i-middle_point);
    points[i]=val + sigma*dist_middle;
    if (points[i] > 2.0 * M_PI){
      points[i]=points[i]-2.0*M_PI;
    }
  }
  for (size_t i = 0; i < middle_point; i++) {
    // std::cout << "low value, " << i << std::endl;
    int dist_middle=abs(i-middle_point);
    points[i]=val - sigma*dist_middle;
    if (points[i]<0){
      points[i]=2.0*M_PI + points[i];
    }

  }

  //first intersections


  // std::cout << "freq is" << freq << std::endl;
  // std::cout << "freq has first dist" << first_dist << std::endl;
  // std::cout << "val is" << val << std::endl;
  for (size_t i = 0; i < num_points; i++) {
    // intersects[0][i]=interpolate(points[i], 0.0 ,2.0*M_PI,0.0,first_dist);

    //previous intersect i think it's bad becaue phase0 should be mapped to the max dist
    intersects[0][i]=interpolate(points[i], 0.0 ,2.0*M_PI, 0.0, first_dist);

    // std::cout << "first intersect is" << intersects[0][i] << std::endl;

    // std::cout << "phase" << points[i] << "has dist" << intersects[0][i] << std::endl;
  }



  for (size_t i = 1; i < num_wraps; i++) {
    for (size_t col = 0; col < num_points; col++) {
      intersects[i][col]=intersects[0][col] + first_dist*i;
      // std::cout << "next intersect"  <<intersects[i][col] << std::endl;
    }
  }

  // std::cout << std::endl << std::endl << std::endl;


  return intersects;
}


string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


//takes an image od floats and normalizes it to 0.0 - 1.0 for visualization
void show_img (Mat img){
  Mat copy;
  img.copyTo(copy);
  double min;
  double max;
  cv::minMaxIdx(img, &min, &max);
  std::cout << "showimg minmax is" << min << " " << max << std::endl;

  cv::Mat adjMap;

  for (size_t i = 0; i < copy.rows; i++) {
     for (size_t j = 0; j < copy.cols; j++) {
       copy.at<float>(i,j)= interpolate (copy.at<float>(i,j), min, max, 0.0, 1.0);
     }
   }

  cv::imshow("Out",copy);
  waitKey(0); //wait infinite time for a keypress

}

void show_img_array (std::vector<Mat> imgs){

  for (size_t i = 0; i < imgs.size(); i++) {
    show_img(imgs[i]);
  }

}





Mat read_file (std::string filename, bool should_interpolate, int clip, int max){



  int rows_ir=424;
  int cols_ir=512;

  Mat image = Mat::zeros(rows_ir,cols_ir, CV_32F);

  //Read txt
  vector <vector <string> > data;
  ifstream infile(filename );
  while (infile) {
    string s;
    if (!getline( infile, s )) break;
    istringstream ss( s );
    vector <string> record;
    while (ss) {
        string s;
        if (!getline( ss, s, ',' )) break;
        record.push_back( s );
        //  std::cout <<  "val is" << atoi(s) << std::endl;
    }

    data.push_back( record );
    }
    if (!infile.eof()){
      cerr << "Fooey!\n";
    }


  for (size_t i = 0; i < rows_ir; i++) {
     for (size_t j = 0; j < cols_ir; j++) {
         float val= stof (data[i][j]);

          // std::cout << "val before is" << val << std::endl;


         //if clip is set we will crop up until that value and then interpolate from that range
         if (clip!=0){
           if (val >clip){
               val=clip;
           }
           if (should_interpolate){
             val=interpolate(val, 0, clip, 0.0, 1.0);
           }
         }else{
           if (should_interpolate){
             val=interpolate(val, 0, max, 0.0, 1.0);
           }

         }
          // std::cout << "val is" << val << std::endl;
          // std::cout << "adding at i and j: " << i << " " << j << " value: " << val << std::endl;
         image.at<float>(i,j)=  val;
        //  std::cout << "added" << image.at<int32_t>(i,j) << std::endl;
     }
   }



   //map the values to 0-255
  //  double min_v;
  //  double max_v;
  //  cv::minMaxIdx(image, &min_v, &max_v);
  //  cv::Mat adjMap;
  //  // expand your range to 0..255. Similar to histEq();
  //  image.convertTo(adjMap,CV_8UC1, 255 / (max_v-min_v), -min_v);
  // //  applyColorMap(adjMap, adjMap, cv::COLORMAP_BONE);
  //   applyColorMap(adjMap, adjMap, cv::COLORMAP_AUTUMN);
  //   cv::imshow("Out",adjMap);
  //   waitKey(0); //wait infinite time for a keypress
  //   // adjMap.convertTo(adjMap,CV_32F);
  //  return adjMap;





  return image;

}



int main(int argc, char* argv[])
{
    int rows_ir=424;
    int cols_ir=512;

    std::vector <Mat> irpic;
    irpic.resize(9);

    Mat image = Mat::zeros(rows_ir,cols_ir, CV_32S);



      //read binary 3
    //   int32_t i;
    //   vector<int32_t> ivector;
    //   char *filename = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/20160616-120945411.ir";
    //   ifstream infile(filename,ios::binary);
    //   if (!infile) {
    //       cerr << "Can't open file " << filename << " for reading" << endl;
    //       exit(EXIT_FAILURE);
    //   }
    //   cout << "Opened binary file " << filename << " for reading." << endl;
    //   while (infile.read(reinterpret_cast<char *>(&i), sizeof(i))) {
    //        ivector.push_back(i);
    //    }
    //   for (unsigned count = 0; count < ivector.size(); count++) {
    //         cout << "ivector[" << count << "] = " << setw(11) << ivector[count] << endl;
    //     }
    //     infile.close();
      //
      //
    //     for (size_t i = 0; i < rows_ir; i++) {
    //            for (size_t j = 0; j < cols_ir; j++) {
    //                image.at<float>(i,j)= std::abs( ivector[i*j]);
    //            }
    //        }
      //


      //READ IRPC IMAGES
//-------------------------------------------------------------------------------------------
      for (size_t i = 0; i < 9; i++) {
          // std::string prefix = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/ir_";
          // std::string prefix = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/wall_data_2/ir_";
          std::string prefix = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/my_room/ir_";
          std::string index = std::to_string(i);
          std::string sufix= ".out";

          std::string filename= prefix + index + sufix;

          std::cout << "reading " << filename << std::endl;
          irpic[i]=read_file(filename,false,0, 32767);  //clip should only used for visualization (value of 500 is retty good)
      }

      std::cout << "applying bilateral filter" << std::endl;

      //bilateral filter the raw voltages
      for (size_t i = 0; i < 9; i++) {
        irpic[i].convertTo(irpic[i],CV_32FC1);
        Mat dest;
        bilateralFilter ( irpic[i], dest, 7, 40, 10 );
        irpic[i]=dest;
      }
      show_img_array(irpic);




      //READ PROC TABLES
//------------------------------------------------------------------------------------------------------
      std::vector <Mat> p0_tables;
      p0_tables.resize(3);
      for (size_t i = 0; i < 3; i++) {
          // std::string prefix = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/p0_";
          std::string prefix = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/my_room/p0_";
          std::string index = std::to_string(i);
          std::string sufix= ".out";

          std::string filename= prefix + index + sufix;

          std::cout << "reading " << filename << std::endl;
          p0_tables[i]=read_file(filename,false,0,65536);  //clip should only used for visualization (value of 500 is retty good)
      }



      //preprocess p0 tables
      for (size_t table = 0; table < 3; table++) {
        // for (size_t i = 0; i < rows_ir; i++) {
        //    for (size_t j = 0; j < cols_ir; j++) {
        //       p0_tables[table].at<float>(i,j)=  - p0_tables[table].at<float>(i,j) * 0.000031 * M_PI;
        //    }
        //  }
       }
      //  show_img_array(p0_tables);
       std::cout << "finished reading raw images" << std::endl;
       // std::cout << "images are type" <<  type2str( irpic[0].type() ) << std::endl;

       //PHASES
//--------------------------------------------------------------------------------------------
      std::vector<Mat> phases;
      for (size_t i = 0; i < 3; i++) {
          Mat image = Mat::zeros(rows_ir,cols_ir, CV_32F);
          phases.push_back(image);
      }
      std::cout << "calculating phasses" << std::endl;


      for (size_t freq_id = 0; freq_id < 3; freq_id++) {
        for (size_t i = 0; i < rows_ir; i++) {
           for (size_t j = 0; j < cols_ir; j++) {

             //loop through the 3 phases and get the voltages
              std::vector<float> volt_touple(3);
              for (size_t phase_id = 0; phase_id < 3; phase_id++) {
                volt_touple[phase_id]=irpic[freq_id*3 + phase_id].at<float>(i,j);
              }


               //get the 3 values of p0
              //  std::vector<float> p0_touple(3);
              //  for (size_t p0_idx = 0; p0_idx < 3; p0_idx++) {
              //    p0_tables[p0_idx].at<float>(i,j)=  - p0_tables[p0_idx].at<float>(i,j) * 0.000031 * M_PI;
              //    p0_touple[p0_idx]=p0_tables[p0_idx].at<float>(i,j);
              //   //  p0_touple[p0_idx]=0.0;
              //  }

               float p_0= - p0_tables[freq_id].at<float>(i,j) * 0.000031 * M_PI;
              //  std::cout << "p0 is" << p_0 << std::endl;
              // p_0=0.0;

               float term_sin= - volt_touple[0] * sin(p_0 + 0)
                                - volt_touple[1] * sin(p_0 + 2.0f*M_PI/3.0f)
                                - volt_touple[2] * sin(p_0 + 4.0f*M_PI/3.0f);
               float term_cos=   volt_touple[0] * cos(p_0 + 0)
                                + volt_touple[1] * cos(p_0+ 2.0f*M_PI/3.0f)
                                + volt_touple[2] * cos(p_0 + 4.0f*M_PI/3.0f);




               float phase_val =atan2 (term_sin, term_cos);

              //  std::cout << "phase is" << phase << std::endl;

              phase_val = phase_val < 0 ? phase_val + M_PI * 2.0f : phase_val;
              phase_val = (phase_val != phase_val) ? 0 : phase_val;

              phases[freq_id].at<float>(i,j)=phase_val ;  //we add pi so that the range is from 0 to 2*pi

           }
         }


      }

      // show_img_array(phases);

      // double min_v;
      // double max_v;
      // cv::minMaxIdx(phases[1], &min_v, &max_v);
      // std::cout << "min is " <<  min_v << std::endl;
      // std::cout << "max is " <<  max_v << std::endl;


      //PHASES WITH EXP
//----------------------------------------------------------------------------------------------------
      Mat phase_exp = Mat::zeros(rows_ir,cols_ir, CV_32F);
      for (size_t phase_id = 0; phase_id < 3; phase_id++) {
        for (size_t i = 0; i < rows_ir; i++) {
           for (size_t j = 0; j < cols_ir; j++) {
             dcomp sum=0.0;

             dcomp imaginary;
             imaginary = -1;
             imaginary = sqrt(imaginary);

             for (int  n = 0; n < 3; n++) {
               sum+=double (irpic[phase_id*3+n].at<float>(i,j)) * exp (-imaginary*(  2*M_PI* n/3 ));
             }

             float phase_val=-arg(sum);
             phase_val = phase_val < 0 ? phase_val + M_PI * 2.0f : phase_val;
             phase_exp.at<float>(i,j)=phase_val;

           }
         }

        //  phases[phase_id]=phase_exp;
        //  phase_exp.copyTo(phases[phase_id]);

      }

      //  show_img(phase_exp);

      // show_img_array(phases);
      // std::cout << "showitn gnew pahses" << std::endl;

      //PHASES FROM libfreenect2
//----------------------------------------------------------------------------------------------------------
      for (size_t i = 0; i < 3; i++) {
          std::string prefix = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/phases_libfreenect2/phase_";
          std::string index = std::to_string(i);
          std::string sufix= ".out";

          std::string filename= prefix + index + sufix;

          std::cout << "reading " << filename << std::endl;
          // phases[i]=read_file(filename,false,0,65536);  //clip should only used for visualization (value of 500 is retty good)
      }

      show_img_array(phases);



      //AMPLITUDES
//----------------------------------------------------------------------------------------------------------
      std::vector<Mat> amplitudes;
      for (size_t i = 0; i < 3; i++) {
          Mat image = Mat::zeros(rows_ir,cols_ir, CV_32F);
          amplitudes.push_back(image);
      }



      for (size_t freq_id = 0; freq_id < 3; freq_id++) {
        for (size_t i = 0; i < rows_ir; i++) {
          for (size_t j = 0; j < cols_ir; j++) {

            //loop through the 3 phases and get the voltages
             std::vector<float> volt_touple(3);
             for (size_t phase_id = 0; phase_id < 3; phase_id++) {
               volt_touple[phase_id]=irpic[freq_id*3 + phase_id].at<float>(i,j);
             }

             std::vector<float> p0_touple(3);
             for (size_t p0_idx = 0; p0_idx < 3; p0_idx++) {
               p0_touple[p0_idx]=p0_tables[p0_idx].at<float>(i,j);
              //  p0_touple[p0_idx]=0.0;
             }


              float term_sin= - volt_touple[0] * sin(p0_touple[0] + 0)
                              - volt_touple[1] * sin(p0_touple[1] + 2*M_PI/3)
                              - volt_touple[2] * sin(p0_touple[2] + 4*M_PI/3);
              float term_cos=   volt_touple[0] * cos(p0_touple[0] + 0)
                               + volt_touple[1] * cos(p0_touple[1]+ 2*M_PI/3)
                               + volt_touple[2] * cos(p0_touple[2] + 4*M_PI/3);

              amplitudes[freq_id].at<float>(i,j)=  pow(term_sin,2) + pow(term_cos,2);
              // amplitudes[freq_id].at<float>(i,j)=  term_sin + term_cos;
              // amplitudes[freq_id].at<float>(i,j)=  sqrt (pow(term_sin,2) + pow(term_cos,2)) /2;









              //again using the code in libfreenect2
              float p0 = -((float) p0_tables[freq_id].at<float>(i,j))* 0.000031 * M_PI;
              // std::cout << "p0 is" << p0 << std::endl;
              // p0=0.0f;

               float tmp0 = p0 + 0.0f;;
               float tmp1 = p0 + 2.094395f;
               float tmp2 = p0 + 4.18879f;

               float cos_tmp0 = std::cos(tmp0);
               float cos_tmp1 = std::cos(tmp1);
               float cos_tmp2 = std::cos(tmp2);

               float sin_negtmp0 = std::sin(-tmp0);
               float sin_negtmp1 = std::sin(-tmp1);
               float sin_negtmp2 = std::sin(-tmp2);
               //
              //  std::cout << "cos_tmp0= " << cos_tmp0 << std::endl;
              //  std::cout << "cos_tmp1= " << cos_tmp1 << std::endl;
              //  std::cout << "cos_tmp2= " << cos_tmp2 << std::endl;
              //  std::cout << "sin_negtmp0= " << sin_negtmp0 << std::endl;
              //  std::cout << "sin_negtmp1= " << sin_negtmp1 << std::endl;
              //  std::cout << "sin_negtmp2= " << sin_negtmp2 << std::endl << std::endl;


              //  std::cout << "tmp2 is " << tmp2 << std::endl;
              //  std::cout << "3= " << cos_tmp2 << std::endl;

               float ir_image_a = cos_tmp0 * volt_touple[0] + cos_tmp1 * volt_touple[1] + cos_tmp2 * volt_touple[2];
               float ir_image_b = sin_negtmp0 * volt_touple[0] + sin_negtmp1 * volt_touple[1] + sin_negtmp2 * volt_touple[2];

               float abMultiplierPerFrq;
              //  float ab_multiplier= 0.6666667f;
              float ab_multiplier= 1.0f;
               if (freq_id==0)
                  abMultiplierPerFrq=1.322581f;
               if (freq_id==1)
                  abMultiplierPerFrq=1.0f;
               if (freq_id==2)
                  abMultiplierPerFrq=1.612903f;

               ir_image_a *= abMultiplierPerFrq;
               ir_image_b *= abMultiplierPerFrq;

               float ir_amplitude = std::sqrt(ir_image_a * ir_image_a + ir_image_b * ir_image_b) * ab_multiplier;

              //  std::cout << "prev value is" << amplitudes[freq_id].at<float>(i,j) << " next one is " << ir_amplitude << std::endl;

               amplitudes[freq_id].at<float>(i,j)=ir_amplitude;


          }
        }
      }


      // show_img_array(amplitudes);

      Mat amplitude_final = Mat::zeros(rows_ir,cols_ir, CV_32F);
      for (size_t i = 0; i < rows_ir; i++) {
        for (size_t j = 0; j < cols_ir; j++) {
          amplitude_final.at<float>(i,j)=  sqrt (amplitudes[0].at<float>(i,j) )
                                         + sqrt (amplitudes[1].at<float>(i,j) )
                                         + sqrt (amplitudes[2].at<float>(i,j) );



        }
      }


      // show_img(amplitude_final);

      //ESTIMATION OF BOX FUNCTION
//--------------------------------------------------------------------------------------------------------
      // cv::Rect roi(306, 160, 95, 1);   //topleft corner x and y   , width and heigh    //box120
      cv::Rect roi(378, 160, 84, 1);
      // cv::Mat image_roi = image(roi)
      //
      //
      Mat phase_2_copy;
      phases[0].copyTo(phase_2_copy);
      cv::rectangle(phase_2_copy, roi, cv::Scalar(7, 0, 0), 1);
      // show_img(phase_2_copy);


      Mat sub_image= phases[0](roi);
      std::cout << "sub_image is type" <<  type2str( sub_image.type() ) << std::endl;
      std::cout << "sub_image is rows, cols" <<  sub_image.rows << " " << sub_image.cols << std::endl;
      // cv::imshow("wat",sub_image);
      // waitKey(0); //wait infinite time for a keypress
      // show_img(sub_image);

      // std::vector<float> box;
      // for (size_t i = 0; i < sub_image.rows; i++) {
      //    for (size_t j = 0; j < sub_image.cols; j++) {
      //      std::cout << "box_val is" << sub_image.at<float>(i,j) << std::endl;
      //      box.push_back(sub_image.at<float>(i,j));
      //    }
      //  }
      // ofstream myfile("box_80.txt");
      // int vsize = box.size();
      // for (int n=0; n<vsize; n++)
      // {
      //     myfile << box[n] << endl;
      // }






      //PRECOMPUTE HYPOTHESIS
//--------------------------------------------------------------------------------------------------
      matrix_type precompute_hypothesis;
      precompute_hypothesis.push_back(std::vector<float>({0,0,0}) );
      precompute_hypothesis.push_back(std::vector<float>({0,0,1}) );
      precompute_hypothesis.push_back(std::vector<float>({1,0,1}) );
      precompute_hypothesis.push_back(std::vector<float>({1,0,2}) );

      precompute_hypothesis.push_back(std::vector<float>({2,0,2}) );
      precompute_hypothesis.push_back(std::vector<float>({2,0,3}) );
      precompute_hypothesis.push_back(std::vector<float>({2,0,4}) );

      precompute_hypothesis.push_back(std::vector<float>({3,0,4}) );
      precompute_hypothesis.push_back(std::vector<float>({3,0,5}) );
      precompute_hypothesis.push_back(std::vector<float>({3,0,6}) );

      precompute_hypothesis.push_back(std::vector<float>({4,0,5}) );
      precompute_hypothesis.push_back(std::vector<float>({4,0,6}) );
      precompute_hypothesis.push_back(std::vector<float>({4,0,7}) );
      precompute_hypothesis.push_back(std::vector<float>({4,1,5}) );
      precompute_hypothesis.push_back(std::vector<float>({4,1,6}) );
      precompute_hypothesis.push_back(std::vector<float>({4,1,7}) );

      precompute_hypothesis.push_back(std::vector<float>({5,0,7}) );
      precompute_hypothesis.push_back(std::vector<float>({5,0,8}) );
      precompute_hypothesis.push_back(std::vector<float>({5,0,9}) );
      precompute_hypothesis.push_back(std::vector<float>({5,1,7}) );
      precompute_hypothesis.push_back(std::vector<float>({5,1,8}) );
      precompute_hypothesis.push_back(std::vector<float>({5,1,9}) );

      precompute_hypothesis.push_back(std::vector<float>({6,1,8}) );
      precompute_hypothesis.push_back(std::vector<float>({6,1,9}) );
      precompute_hypothesis.push_back(std::vector<float>({6,1,10}) );

      precompute_hypothesis.push_back(std::vector<float>({7,1,10}) );
      precompute_hypothesis.push_back(std::vector<float>({7,1,11}) );
      precompute_hypothesis.push_back(std::vector<float>({7,1,12}) );

      precompute_hypothesis.push_back(std::vector<float>({8,1,11}) );
      precompute_hypothesis.push_back(std::vector<float>({8,1,12}) );
      precompute_hypothesis.push_back(std::vector<float>({8,1,13}) );

      precompute_hypothesis.push_back(std::vector<float>({9,1,13}) );
      precompute_hypothesis.push_back(std::vector<float>({9,1,14}) );

      // for (size_t i = 0; i < 33; i++) {
      //   std::cout << " value is" << precompute_hypothesis[i][0] << " " << precompute_hypothesis[i][1] << " " <<  precompute_hypothesis[i][2] << std::endl;
      // }
      // std::cout << "number of hyptoshes " << precompute_hypothesis.size() << std::endl;


      //UNWRAPPING USING THE TEXT MINIMZATION
//----------------------------------------------------------------------------------------------
      std::cout << "begginign unwrapping" << std::endl;
      Mat image_n_0 = Mat::zeros(rows_ir,cols_ir, CV_32F);
      Mat image_n_1 = Mat::zeros(rows_ir,cols_ir, CV_32F);
      Mat image_n_2 = Mat::zeros(rows_ir,cols_ir, CV_32F);
      for (size_t i = 0; i < rows_ir; i++) {
         for (size_t j = 0; j < cols_ir; j++) {

           float t_0= 3.0*  phases[0].at<float>(i,j) / (2*M_PI);
           float t_1= 15.0* phases[1].at<float>(i,j) / (2*M_PI);
           float t_2= 2.0*  phases[2].at<float>(i,j) / (2*M_PI);

           float sigma_phi_0 = 1.0/80.0;
           float sigma_phi_1 = 1.0/16.0;
           float sigma_phi_2 = 1.0/120.0;

          // float sigma_phi_0 = 1.0/1.0;
          // float sigma_phi_1 = 1.0/1.0;
          // float sigma_phi_2 = 1.0/1.0;

           float sigma_t_0= 3.0* sigma_phi_0/(2.0*M_PI);
           float sigma_t_1= 15.0*sigma_phi_1/(2.0*M_PI);
           float sigma_t_2= 2.0* sigma_phi_2/(2.0*M_PI);
           //
           float sigma_epsilon_1=  sigma_t_1*sigma_t_1 + sigma_t_0*sigma_t_0;
           float sigma_epsilon_2=  sigma_t_2*sigma_t_2 + sigma_t_0*sigma_t_0;
           float sigma_epsilon_3=  sigma_t_2*sigma_t_2 + sigma_t_1*sigma_t_1;

          //  float sigma_epsilon_1=sqrt(1.0/0.7007);
          //  float sigma_epsilon_2=sqrt(1.0/366.2946);
          //  float sigma_epsilon_3=sqrt(1.0/0.7016);


          //  float min_cost=1000000000000.0;
           float min_cost=std::numeric_limits<float>::max();

           int best_n_0=100;
           int best_n_1=100;
           int best_n_2=100;


           for (int n_0 = 0; n_0 < 10; n_0++) {
             for (int n_1 = 0; n_1 < 1; n_1++) {
               for (int n_2 = 0; n_2 < 15; n_2++) {
                 float epislon_1 = 3.0*n_0 - 15.0* n_1 - (t_1 - t_0);
                 float epsilon_2 = 3.0*n_0 - 2.0*n_2   - (t_2 - t_0);
                 float epsilon_3 = 15.0*n_1 - 2.0*n_2  - (t_2 - t_1);

                 float cost= pow(epislon_1, 2)/ pow(sigma_epsilon_1,2)+
                              pow(epsilon_2, 2)/ pow(sigma_epsilon_2,2)+
                              pow(epsilon_3, 2)/ pow(sigma_epsilon_3,2);

                    // std::cout << "cost is" << cost << std::endl;

                  if (cost < min_cost){
                    min_cost=cost;
                    best_n_0=n_0;
                    best_n_1=n_1;
                    best_n_2=n_2;
                  }

               }
             }
           }



          // for (size_t hyp = 0; hyp < precompute_hypothesis.size(); hyp++) {
          //   int n_0= precompute_hypothesis[hyp][0];
          //   int n_1= precompute_hypothesis[hyp][1];
          //   int n_2= precompute_hypothesis[hyp][2];
          //
          //
          //
          //     float epislon_1 = 3.0*n_0 - 15.0* n_1 - (t_1 - t_0);
          //      float epsilon_2 = 3.0*n_0 - 2.0*n_2   - (t_2 - t_0);
          //      float epsilon_3 = 15.0*n_1 - 2.0*n_2  - (t_2 - t_1);
          //
          //      float cost= pow(epislon_1, 2)/ pow(sigma_epsilon_1,2)+
          //                   pow(epsilon_2, 2)/ pow(sigma_epsilon_2,2)+
          //                   pow(epsilon_3, 2)/ pow(sigma_epsilon_3,2);
          //
          //           // std::cout << "cost is" << cost << std::endl;
          //
          //       if (cost < min_cost){
          //         min_cost=cost;
          //         best_n_0=n_0;
          //         best_n_1=n_1;
          //         best_n_2=n_2;
          //       }
          //
          //
          // }

              // std::cout << "best is " << best_n_0 << " " << best_n_1 << " " << best_n_2 << std::endl;

           image_n_0.at<float>(i,j)=best_n_0;
           image_n_1.at<float>(i,j)=best_n_1;
           image_n_2.at<float>(i,j)=best_n_2;

         }
       }
       std::cout << "finished unwrapping" << std::endl;
       //
       //
      //  show_img(image_n_0);
      //  show_img(image_n_1);
      //  show_img(image_n_2);
      //  //


       //scale them
       Mat phase_0_scaled = Mat::zeros(rows_ir,cols_ir, CV_32F);
       for (size_t i = 0; i < rows_ir; i++) {
          for (size_t j = 0; j < cols_ir; j++) {
            phase_0_scaled.at<float>(i,j)= 3* (phases[0].at<float>(i,j) + 2*M_PI*image_n_0.at<float>(i,j)) /  (2 * M_PI);
          }
        }

      Mat phase_1_scaled = Mat::zeros(rows_ir,cols_ir, CV_32F);
      for (size_t i = 0; i < rows_ir; i++) {
         for (size_t j = 0; j < cols_ir; j++) {
           phase_1_scaled.at<float>(i,j)=  15* ( phases[1].at<float>(i,j) + 2*M_PI*image_n_1.at<float>(i,j) ) / (2 * M_PI);
         }
       }

       Mat phase_2_scaled = Mat::zeros(rows_ir,cols_ir, CV_32F);
       for (size_t i = 0; i < rows_ir; i++) {
          for (size_t j = 0; j < cols_ir; j++) {
            phase_2_scaled.at<float>(i,j)=2 * ( phases[2].at<float>(i,j) + 2*M_PI*image_n_2.at<float>(i,j) ) / (2 * M_PI);
          }
        }


      Mat phase_fused = Mat::zeros(rows_ir,cols_ir, CV_32F);
      for (size_t i = 0; i < rows_ir; i++) {
         for (size_t j = 0; j < cols_ir; j++) {
           float sigma_phi_0 = 1.0/80.0;
           float sigma_phi_1 = 1.0/16.0;
           float sigma_phi_2 = 1.0/120.0;

           float sigma_t_0= 3.0* sigma_phi_0/(2.0*M_PI);
           float sigma_t_1= 15.0*sigma_phi_1/(2.0*M_PI);
           float sigma_t_2= 2.0* sigma_phi_2/(2.0*M_PI);

           float sum = phase_0_scaled.at<float>(i,j) / (sigma_t_0) +
                        phase_1_scaled.at<float>(i,j) / (sigma_t_1)+
                        phase_2_scaled.at<float>(i,j) / (sigma_t_2);

          float normalizer=1 / ( 1/ sigma_t_0 + 1/sigma_t_1 + 1/sigma_t_2);
          phase_fused.at<float>(i,j)= normalizer*sum;
         }
       }


      // show_img(phase_0_scaled);
      // show_img(phase_1_scaled);
      // show_img(phase_2_scaled);

      // show_img(phase_fused);














      //SAWTOOTH METHOD
//---------------------------------------------------------------------------------------------------------

      int num_points=1;
      int num_wraps_freq_1=10;
      int num_wraps_freq_2=2;
      int num_wraps_freq_3=15;

      float sigma_80=  0.05;
      float sigma_16=  0.1;
      float sigma_120= 0.001;

      Mat confidence_mat = Mat::zeros(rows_ir,cols_ir, CV_32F);
      Mat mean_dist_mat = Mat::zeros(rows_ir,cols_ir, CV_32F);

      for (size_t i = 0; i < rows_ir; i++) {
         for (size_t j = 0; j < cols_ir; j++) {
            // val_80 =phases[row, col, 0]
            // val_16 =phases[row, col, 1]
            // val_120=phases[row, col, 2]
            // intersections_80 = intersect_saw  (80, num_wraps_freq_1, val_80, num_points, sigma_80)
            // intersections_16 = intersect_saw  (16, num_wraps_freq_2, val_16, num_points, sigma_16)
            // intersections_120= intersect_saw (120, num_wraps_freq_3, val_120, num_points, sigma_120)

            float val_80= phases[0].at<float>(i,j);
            float val_16= phases[1].at<float>(i,j);
            float val_120= phases[2].at<float>(i,j);

            matrix_type intersections_80 = intersect_saw  (80, num_wraps_freq_1, val_80, num_points, sigma_80);
            matrix_type intersections_16 = intersect_saw  (16, num_wraps_freq_2, val_16, num_points, sigma_16);
            matrix_type intersections_120= intersect_saw  (120, num_wraps_freq_3, val_120, num_points, sigma_120);

            int num_rows_80 = intersections_80.size();
            int num_cols_80 = intersections_80[0].size();

            int num_rows_16 = intersections_16.size();
            int num_cols_16 = intersections_16[0].size();

            int num_rows_120 = intersections_120.size();
            int num_cols_120 = intersections_120[0].size();

            float difference=std::numeric_limits<float>::max();;
            int n_0=100;
            int n_1=100;
            int n_2=100;
            float mean_dist=100.0;

            // std::cout << "rows, cols for 80 " <<  num_rows_80 << " " << num_cols_80 << std::endl;


            for (size_t row_80 = 0; row_80 < num_rows_80; row_80++) {
              for (size_t col_80 = 0; col_80 < num_cols_80; col_80++) {
                for (size_t row_16 = 0; row_16 < num_rows_16; row_16++) {
                  for (size_t col_16 = 0; col_16 < num_cols_16; col_16++) {
                    for (size_t row_120 = 0; row_120 < num_rows_120; row_120++) {
                      for (size_t col_120 = 0; col_120 < num_cols_120; col_120++) {
                        float dif_0=abs(intersections_80[row_80][col_80]-intersections_16[row_16][col_16])
                                    *(1.0/(1.0/80.0 + 1.0/16.0));
                        float dif_1=abs(intersections_80[row_80][col_80]-intersections_120[row_120][col_120])
                                    *(1.0/(1.0/80.0 + 1.0/120.0));
                        float dif_2=abs(intersections_120[row_120][col_120]-intersections_16[row_16][col_16])
                                    *(1.0/(1.0/120.0 + 1.0/16.0));
                        float dif_final=(dif_0+dif_1+dif_2);

                        if (dif_final < difference){
                          difference=dif_final;
                          mean_dist=(intersections_80[row_80][col_80] +intersections_16[row_16][col_16]+ intersections_120[row_120][col_120])/3;

                          n_0= floor (intersections_80[row_80][col_80] / (LIGHT * (2*M_PI) /(4*M_PI*80*1000000) ) );
                          n_1= floor (intersections_16[row_16][col_16]/ (LIGHT * (2*M_PI) /(4*M_PI*16*1000000) ) );
                          n_2= floor (intersections_120[row_120][col_120]/ (LIGHT * (2*M_PI) /(4*M_PI*120*1000000) ) );
                        }
//                                 difference=dif_final
// #                                        print "dif_final is", dif_final
// #                                        mean_dist=(intersections_80[row_80,col_80] +intersections_16[row_16,col_16]+ intersections_120[row_120,col_120])/3
//                                         mean_dist=(intersections_80[row_80,col_80])
//                                         n_0= floor (intersections_80[row_80,col_80]/ (csts.c * (2*np.pi) /(4*np.pi*80*1000000) ) )
//                                         n_1= floor (intersections_16[row_16,col_16]/ (csts.c * (2*np.pi) /(4*np.pi*16*1000000) ) )
//                                         n_2= floor (intersections_120[row_120,col_120]/ (csts.c * (2*np.pi) /(4*np.pi*120*1000000) ) )

                      }
                    }
                  }
                }
              }
            }

            // std::cout << "n_1" << n_1 << std::endl;

            confidence_mat.at<float>(i,j)=difference;
            image_n_0.at<float>(i,j)=n_0;
            image_n_1.at<float>(i,j)=n_1;
            image_n_2.at<float>(i,j)=n_2;
            mean_dist_mat.at<float>(i,j)=mean_dist;

            // std::cout << "best is " << n_0 << " " << n_1 << " " << n_2 << std::endl;

         }
       }

    //  double min_v;
    //  double max_v;
    //  cv::minMaxIdx(confidence_mat, &min_v, &max_v);
    //  std::cout << "min is " <<  min_v << std::endl;
    //  std::cout << "max is " <<  max_v << std::endl;
    show_img(confidence_mat);
    show_img(image_n_0);
    show_img(image_n_1);
    show_img(image_n_2);
    // show_img(mean_dist_mat);





    //scaled phases
    std::vector<Mat> scaled_phases;
    for (size_t i = 0; i < 3; i++) {
        Mat image = Mat::zeros(rows_ir,cols_ir, CV_32F);
        scaled_phases.push_back(image);
    }

    Mat coeficients;
    for (size_t phase_id = 0; phase_id < 3; phase_id++) {
      for (size_t i = 0; i < rows_ir; i++) {
        for (size_t j = 0; j < cols_ir; j++) {
          if (phase_id==0)
            coeficients=image_n_0;
          if (phase_id==1)
            coeficients=image_n_1;
          if (phase_id==2)
            coeficients=image_n_2;

          scaled_phases[phase_id].at<float>(i,j)= phases[phase_id].at<float>(i,j) +  coeficients.at<float>(i,j)* 2*M_PI;
        }
      }
    }


    // show_img_array(scaled_phases);



    for (size_t i = 0; i < rows_ir; i++) {
       for (size_t j = 0; j < cols_ir; j++) {
         float sigma_phi_0 = 1.0/80.0;
         float sigma_phi_1 = 1.0/16.0;
         float sigma_phi_2 = 1.0/120.0;

         float sigma_t_0= 3.0* sigma_phi_0/(2.0*M_PI);
         float sigma_t_1= 15.0*sigma_phi_1/(2.0*M_PI);
         float sigma_t_2= 2.0* sigma_phi_2/(2.0*M_PI);

         float sum = scaled_phases[0].at<float>(i,j) / (sigma_t_0) +
                      scaled_phases[1].at<float>(i,j) / (sigma_t_1)+
                      scaled_phases[2].at<float>(i,j) / (sigma_t_2);

        float normalizer=1 / ( 1/ sigma_t_0 + 1/sigma_t_1 + 1/sigma_t_2);
        phase_fused.at<float>(i,j)= normalizer*sum;
       }
     }

     show_img(phase_fused);





     //see how many points are at maximum value, because those are bad
    //   double min_v;
    //   double max_v;
    //   cv::minMaxIdx(phase_fused, &min_v, &max_v);
    //   std::cout << "min is " <<  min_v << std::endl;
    //   std::cout << "max is " <<  max_v << std::endl;
    //   int count_bad=0;
    //  for (size_t i = 0; i < rows_ir; i++) {
    //     for (size_t j = 0; j < cols_ir; j++) {
    //       if (phase_fused.at<float>(i,j)>80.0){
    //         count_bad++;
    //       }
    //     }
    //   }
    //   std::cout << "bad pixels are" << count_bad << std::endl;



      //median blur depending on confidence_mat
      double min_confidence;
      double max_confidence;
      cv::minMaxIdx(confidence_mat, &min_confidence, &max_confidence);
      std::cout << "min is " <<  min_confidence << std::endl;
      std::cout << "max is " <<  max_confidence << std::endl;
      Mat phase_fused_filtered = Mat::zeros(rows_ir,cols_ir, CV_32F);




      for (size_t i = 0; i < rows_ir; i++) {
         for (size_t j = 0; j < cols_ir; j++) {





          //  std::cout << " pixel at " << i << " " << j << " val= " << confidence_mat.at<float>(i,j) << std::endl;
           if (  (confidence_mat.at<float>(i,j) > 0.0f) &&  ( confidence_mat.at<float>(i,j) <  (max_confidence*(1.0/3.0))) ){
             //good value
            //  std::cout << " good value" << std::endl;
             phase_fused_filtered.at<float>(i,j)= phase_fused.at<float>(i,j);


           }else if ( (confidence_mat.at<float>(i,j) > (max_confidence*(1.0/3.0)) ) &&  (confidence_mat.at<float>(i,j) <  (max_confidence*(2.0/3.0)))){
             //blur a little bit
            //  std::cout << " blur a little bit" << std::endl;

             int window_size=7;
             int step_size= window_size/2;
             std::vector<float> values;

             for (int w_row = -step_size; w_row < step_size; w_row++) {
               for (int w_col = -step_size; w_col < step_size; w_col++) {

                 int index_row=i+w_row;
                 int index_col=j+w_col;

                 if (index_row<0 || index_row>rows_ir || index_col<0 || index_col>cols_ir){
                   values.push_back(0.0f);
                 }else{
                   //valid pixel, push back that value
                   values.push_back (   phase_fused.at<float>(index_row,index_col)  );
                 }


               }
             }




               std::vector<float> calc(values);
               size_t n = calc.size() / 2;
               std::nth_element(calc.begin(), calc.begin()+n, calc.end());
               double median= calc[n];
              //  std::cout << "median is:" << median << std::endl;
               phase_fused_filtered.at<float>(i,j)=median;



           }else if ( ( confidence_mat.at<float>(i,j) > (max_confidence*(2.0/3.0)))){
             //blur more
            //  std::cout << " blur a more" << std::endl;


             int window_size=13;
             int step_size= window_size/2;
             std::vector<float> values;

            //  std::cout << "step_size is" << step_size << std::endl;

             for (int w_row = -step_size; w_row < step_size; w_row++) {
               for (int w_col = -step_size; w_col < step_size; w_col++) {
                //  std::cout << "loop" << std::endl;

                 int index_row=i+w_row;
                 int index_col=j+w_col;

                 if (index_row<0 || index_row>rows_ir || index_col<0 || index_col>cols_ir){
                   values.push_back(0.0f);
                 }else{
                   //valid pixel, push back that value
                   values.push_back (   phase_fused.at<float>(index_row,index_col)  );
                 }


               }
             }


            //  std::cout<< "finihed blurring more" << std::endl;
            //  std::cout << "values has size" << values.size() << std::endl;

               std::vector<float> calc(values);
               size_t n = calc.size() / 2;
               std::nth_element(calc.begin(), calc.begin()+n, calc.end());
               double median= calc[n];
              //  std::cout << "median is:" << median << std::endl;
               phase_fused_filtered.at<float>(i,j)=median;

           }




         }
       }
       show_img(phase_fused_filtered);



       //For better visualization interpolat from min to max-> min-median
       Mat phase_fused_interp = Mat::zeros(rows_ir,cols_ir, CV_32F);
       std::vector<float> values;
       for (size_t i = 0; i < rows_ir; i++) {
          for (size_t j = 0; j < cols_ir; j++) {
             values.push_back (   phase_fused_filtered.at<float>(i,j)  );
          }
        }
        std::vector<float> calc(values);
        size_t n = calc.size() / 2;
        std::nth_element(calc.begin(), calc.begin()+n, calc.end());
        double median= calc[n];


        double min_phase_filtered;
        double max_phase_filtered;
        cv::minMaxIdx(confidence_mat, &min_phase_filtered, &max_phase_filtered);
        std::cout << "min is " <<  min_phase_filtered << std::endl;
        std::cout << "max is " <<  max_phase_filtered << std::endl;
        std::cout << "median is " <<  median << std::endl;


        for (size_t i = 0; i < rows_ir; i++) {
           for (size_t j = 0; j < cols_ir; j++) {
              phase_fused_interp.at<float>(i,j)= std::min (phase_fused_filtered.at<float>(i,j), float(median)*3 );
           }
         }
         show_img(phase_fused_interp);




//TODO p0 table is for each frequncy.  po_0 ->freq_80   p0_1->freq_16

    return 0;
}
