import 'package:flutter/material.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:verifact_app/extensions/context_extensions.dart';
import 'package:verifact_app/utils/constants/sizes.dart';

class CustomSourceCard extends StatelessWidget {
  const CustomSourceCard({
    required this.title,
    required this.url,
    required this.score,
    required this.text,
    super.key,
  });

  final String title;
  final String text;
  final String url;
  final double score;

  Future<void> _launchUrl() async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    // derive a small logo url for known domains (use Google's favicon service)
    String? _logoUrl() {
      try {
        final host = Uri.parse(url).host.toLowerCase();
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
    return Card(
      margin: EdgeInsets.zero,
      color: context.color.surface,
      elevation: 0.5,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppSizes.borderRadiusLg),
        side: BorderSide(color: context.color.outlineVariant),
      ),
      child: InkWell(
        onTap: _launchUrl,
        borderRadius: BorderRadius.circular(AppSizes.borderRadiusLg),
        child: Padding(
          padding: EdgeInsets.all(AppSizes.md),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (logo != null) ...[
                Container(
                  height: 28,
                  width: 28,
                  margin: EdgeInsets.only(right: AppSizes.smMd),
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
                      children: [
                        Expanded(
                          child: Text(
                            title,
                            style: context.text.titleLarge?.copyWith(
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
                      url,
                      style: context.text.labelSmall?.copyWith(
                        color: context.color.secondary,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    SizedBox(height: AppSizes.xsSm),
                    Text(
                      text,
                      style: context.text.bodySmall?.copyWith(),
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                    ),
                    SizedBox(height: AppSizes.xsSm),
                    Text(
                      'Score ${score.toStringAsFixed(2)}',
                      style: context.text.labelSmall?.copyWith(
                        color: context.color.primary,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
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
