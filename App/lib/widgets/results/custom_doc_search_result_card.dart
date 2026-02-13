import 'package:flutter/material.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class CustomDocSearchResultCard extends StatelessWidget {
  const CustomDocSearchResultCard({
    required this.title,
    required this.content,
    required this.score,
    required this.author,
    required this.url,
    super.key,
  });

  final String title;
  final String content;
  final double? score;
  final String author;
  final String? url;

  @override
  Widget build(BuildContext context) {
    final borderRadius = BorderRadius.circular(AppSizes.borderRadiusLg);

    // derive a small logo url for known domains (use Google's favicon service)
    String? _logoUrl() {
      try {
        if (url == null) return null;
        final host = Uri.parse(url!).host.toLowerCase();
        if (host.contains('webmd')) {
          return 'https://www.google.com/s2/favicons?domain=webmd.com&sz=128';
        }
        if (host.contains('who') || host.contains('who.int')) {
          return 'https://www.google.com/s2/favicons?domain=who.int&sz=128';
        }
      } catch (_) {}
      return null;
    }

    final logo = _logoUrl();

    return GestureDetector(
      onTap: () async {
        // if Pressed, it should open the url in the browser
        if (url != null) {
          final uri = Uri.parse(url!);
          await launchUrl(uri, mode: LaunchMode.externalApplication);
        }
      },
      child: Card(
        margin: EdgeInsets.zero,
        color: context.color.surface,
        elevation: 0.5,
        shape: RoundedRectangleBorder(
          borderRadius: borderRadius,
          side: BorderSide(color: context.color.outlineVariant),
        ),
        child: Padding(
          padding: EdgeInsets.all(AppSizes.md),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (logo != null) ...[
                Container(
                  height: 28,
                  width: 28,
                  margin: EdgeInsets.only(right: AppSizes.md),
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                  ),
                  child: ClipOval(
                    child: Image.network(
                      logo,
                      fit: BoxFit.cover,
                      errorBuilder: (_, _, _) => Icon(
                        LucideIcons.link,
                        size: AppSizes.iconLg,
                        color: context.color.onSurfaceVariant,
                      ),
                    ),
                  ),
                ),
              ],
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: Text(
                            title,
                            style: context.text.titleMedium?.copyWith(
                              fontWeight: FontWeight.w700,
                              color: context.color.onSurface,
                            ),
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        SizedBox(width: AppSizes.sm),
                        Icon(
                          LucideIcons.externalLink,
                          size: AppSizes.iconXs,
                          color: context.color.secondary,
                        ),
                      ],
                    ),
                    Text(
                      'By $author',
                      style: context.text.labelSmall?.copyWith(
                        color: context.color.tertiary,
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                    SizedBox(height: AppSizes.xsSm),
                    Text(
                      content,
                      style: context.text.bodySmall?.copyWith(
                        color: context.color.onSurfaceVariant,
                      ),
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                    ),
                    if (score != null) ...[
                      SizedBox(height: AppSizes.xsSm),
                      Text(
                        'Score ${score!.toStringAsFixed(2)}',
                        style: context.text.labelSmall?.copyWith(
                          color: context.color.primary,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
