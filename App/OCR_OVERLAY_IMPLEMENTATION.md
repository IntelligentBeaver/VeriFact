# OCR Text Box Overlay Implementation

## Overview
This document describes the implementation of a text box overlay system on the OCR screen that displays the detected text regions from MLkit's text recognition output.

## Changes Made

### 1. **OcrProvider** (`lib/providers/ocr_provider.dart`)

#### New Data Models
- **`TextBlock` class**: Represents a recognized text region with:
  - `text`: The recognized text content
  - `boundingBox`: A `Rect` object with normalized coordinates (0-1) representing the text region's position relative to the image

#### Updated `OcrState`
- Added `textBlocks: List<TextBlock>` - stores all detected text regions
- Added `selectedBlockIndex: int?` - tracks which text block is currently selected

#### New Methods in `OcrNotifier`
- `selectTextBlock(int? index)` - selects a text block by index
- `getSelectedTextBlock()` - returns the currently selected text block
- `retakeImage()` - resets the OCR state to allow capturing a new image

### 2. **OcrService** (`lib/services/ocr_service.dart`)

#### Updated Imports
- Added MLkit import with alias: `import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart' as ml_kit;`
- This avoids naming conflicts with the custom `TextBlock` class

#### Updated `OcrPickerResult`
- Now includes `textBlocks: List<TextBlock>` field

#### New Methods
- `recognizeTextWithBlocks(XFile imageFile)` - processes an image and extracts text blocks with bounding boxes from MLkit
- Returns a `List<TextBlock>` containing all detected text regions

#### Updated Methods
- `pickImageAndRecognizeText()` - now calls both `recognizeText()` and `recognizeTextWithBlocks()` to return complete results

### 3. **OcrScreen** (`lib/screens/ocr_page.dart`)

#### New Custom Painter
- **`TextBoxOverlayPainter`**: Custom Flutter painter that:
  - Draws rectangles for each text block
  - Uses blue with 100 alpha for unselected boxes
  - Uses red with 150 alpha for the selected box with thicker border
  - Adds a green fill (30 alpha) behind the selected box for visibility
  - Displays a text label above the selected box showing a preview of the recognized text

#### Enhanced UI Components

##### Image Display with Overlay (`_buildImageWithOverlay`)
- Stack widget combining:
  - Original image
  - Interactive overlay using `CustomPaint` with `TextBoxOverlayPainter`
  - `GestureDetector` for tap detection on text boxes

##### Text Blocks List (`_buildTextBlocksList`)
- Horizontal scrollable list of all detected text blocks
- Clickable cards showing text preview
- Selected block highlighted in blue
- Click to toggle selection

##### Updated Display Area (`_buildDisplayArea`)
- Passes both state and notifier to child methods
- Displays the image with overlay
- Shows the text blocks list
- Displays full recognized text
- Copy and Retake buttons

#### Tap Handling
- `_selectTextBoxAtPosition()` method handles tap detection on the image
- Maps tap coordinates to detect which text block was selected
- Integrates with the horizontal list for alternative selection

## How It Works

### Flow:
1. **User picks image** → `pickAndRecognizeImage()` is called
2. **MLkit processes image** → Extracts text and bounding box information
3. **Text blocks extracted** → Bounding boxes are converted to relative coordinates (0-1)
4. **UI updates** → Screen displays:
   - Image with text box overlay
   - Horizontal list of detected text blocks
   - Full recognized text
5. **User can select box** → Either:
   - Click on the box directly in the image
   - Click on a block in the horizontal list
6. **Visual feedback** → Selected box is highlighted with red border and green fill

## Bounding Box Coordinate System

- MLkit returns bounding boxes in **pixel coordinates** relative to the image
- The implementation converts these to **normalized coordinates (0-1)** for display
- This allows the overlay to work correctly regardless of image size
- During rendering, normalized coordinates are multiplied by the canvas size

## Customization Options

You can modify the appearance by changing:
- **Colors**: In `TextBoxOverlayPainter.paint()`
  - `Colors.blue` for unselected boxes
  - `Colors.red` for selected box
  - `Colors.green` for fill
- **Stroke width**: Adjust `strokeWidth` values
- **Alpha values**: Modify the `withAlpha()` values for transparency
- **Text label size**: Change `fontSize` in `_drawTextLabel()`

## Future Enhancements

Potential improvements:
1. **Copy single block** - Add ability to copy individual text block text
2. **Block reordering** - Allow users to manually adjust bounding boxes
3. **Language detection** - Show detected language for each block
4. **Confidence scores** - Display MLkit's confidence level for each block
5. **Edit mode** - Allow users to correct OCR mistakes inline
6. **Export** - Export bounding boxes with coordinates for data processing

## Testing

To test the implementation:
1. Run the app and navigate to the OCR screen
2. Pick an image from gallery or camera
3. Verify that text boxes appear as overlays on the image
4. Click on different boxes to see them highlight in red
5. Verify the text blocks list shows all detected regions
6. Check that the copy functionality works with the recognized text

## Files Modified

- `lib/providers/ocr_provider.dart` - State management and models
- `lib/services/ocr_service.dart` - MLkit integration and text extraction
- `lib/screens/ocr_page.dart` - UI and visualization
