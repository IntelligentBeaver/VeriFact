import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class AppSizes {
  AppSizes._();

  //* Padding and Margin Sizes
  static double get xs => 4.w;
  static double get xsSm => 6.w;
  static double get sm => 8.w;
  static double get smMd => 12.w;
  static double get md => 16.w;
  static double get mdLg => 20.w;
  static double get lg => 24.w;
  static double get lgXl => 28.w;
  static double get xl => 32.w;

  //* Icon sizes
  static double get iconXs => 12.r;
  static double get iconSm => 16.r;
  static double get iconMd => 24.r;
  static double get iconLg => 32.r;
  static double get iconXl => 38.r;

  //* Font sizes
  static double get label => 14.sp;
  static double get body => 16.sp;
  static double get title => 18.sp;
  static double get headline => 20.sp;
  static double get displaySm => 24.sp;
  static double get displayMd => 28.sp;
  static double get displayLg => 32.sp;

  //* Button sizes
  static double get buttonHeight => 18.h;
  static double get buttonRadius => 12.r;
  static double get buttonWidth => 120.w;
  static double get buttonElevation => 4.r;

  //* AppBar height
  static double get appBarHeight => 56.h;

  //* Image sizes
  static double get imageThumbSize => 89.h;

  //* Default spacing between sections
  static double get defaultSpace => lg;
  static double get spaceBtwSection => md;
  static double get spaceBtwItems => smMd;
  static double get spaceBtwSectionLg => xl;

  //* Responsive Vertical Spacings
  static Widget get verticalXs => 4.verticalSpace;
  static Widget get verticalXsSm => 6.verticalSpace;
  static Widget get verticalSm => 8.verticalSpace;
  static Widget get verticalSmMd => 12.verticalSpace;
  static Widget get verticalMd => 16.verticalSpace;
  static Widget get verticalMdLg => 20.verticalSpace;
  static Widget get verticalLg => 24.verticalSpace;
  static Widget get verticalLgXl => 28.verticalSpace;
  static Widget get verticalXl => 32.verticalSpace;

  //* Responsive Vertical Spacings
  static Widget get horizontalXs => 4.horizontalSpace;
  static Widget get horizontalXsSm => 6.horizontalSpace;
  static Widget get horizontalSm => 8.horizontalSpace;
  static Widget get horizontalSmMd => 12.horizontalSpace;
  static Widget get horizontalMd => 16.horizontalSpace;
  static Widget get horizontalMdLg => 20.horizontalSpace;
  static Widget get horizontalLg => 24.horizontalSpace;
  static Widget get horizontalLgXl => 28.horizontalSpace;
  static Widget get horizontalXl => 32.horizontalSpace;

  //* Border Radius
  static double get borderRadiusXs => 2.r;
  static double get borderRadiusSm => 4.r;
  static double get borderRadiusMd => 8.r;
  static double get borderRadiusLg => 12.r;
  static double get borderRadiusXl => 18.r;

  //* Divider height
  static double get dividerHeight => 1.h;

  //* Input field
  static double get inputFieldRadius => 12.r;
}
