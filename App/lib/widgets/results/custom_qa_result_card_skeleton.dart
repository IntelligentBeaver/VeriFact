import 'package:flutter/material.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class CustomQAResultCardSkeleton extends StatelessWidget {
  const CustomQAResultCardSkeleton({super.key});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.zero,
      color: context.color.surface,
      elevation: 0.5,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppSizes.borderRadiusLg),
        side: BorderSide(color: context.color.outlineVariant),
      ),
      child: Padding(
        padding: EdgeInsets.all(AppSizes.md),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              height: 12.0,
              width: 80.0,
              color: context.color.outlineVariant,
            ),
            SizedBox(height: AppSizes.xsSm),
            Container(
              height: 16.0,
              color: context.color.outlineVariant,
            ),
            SizedBox(height: AppSizes.md),
            Container(
              height: 12.0,
              width: 70.0,
              color: context.color.outlineVariant,
            ),
            SizedBox(height: AppSizes.xsSm),
            Container(
              height: 14.0,
              color: context.color.outlineVariant,
            ),
          ],
        ),
      ),
    );
  }
}
