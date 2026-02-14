import 'package:flutter/material.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class CustomQAResultCard extends StatelessWidget {
  const CustomQAResultCard({
    required this.conclusion,
    required this.evidence,
    super.key,
  });

  final String conclusion;
  final String evidence;

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
            // Conclusion Section
            Text(
              'Conclusion',
              style: context.text.labelMedium?.copyWith(
                fontWeight: FontWeight.w700,
                color: context.color.onSurfaceVariant,
              ),
            ),
            SizedBox(height: AppSizes.xsSm),
            Text(
              conclusion,
              style: context.text.titleLarge?.copyWith(
                color: context.color.onSurface,
              ),
            ),
            SizedBox(height: AppSizes.md),
            // Evidence Section
            Text(
              'Evidence',
              style: context.text.labelMedium?.copyWith(
                fontWeight: FontWeight.w700,
                color: context.color.onSurfaceVariant,
              ),
            ),
            SizedBox(height: AppSizes.xsSm),
            Text(
              evidence,
              style: context.text.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
