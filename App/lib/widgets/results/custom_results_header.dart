import 'package:flutter/material.dart';
import 'package:verifact_app/extensions/context_extensions.dart';

class CustomResultsHeader extends StatelessWidget {
  const CustomResultsHeader({
    required this.title,
    this.onReset,
    super.key,
  });

  final String title;
  final VoidCallback? onReset;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          title,
          style: context.text.displayMedium?.copyWith(
            fontWeight: FontWeight.w900,
            color: context.color.onSurface,
          ),
        ),
        if (onReset != null)
          TextButton(
            onPressed: onReset,
            child: Text(
              'Reset',
              style: context.text.labelMedium?.copyWith(
                color: context.color.primary,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
      ],
    );
  }
}
