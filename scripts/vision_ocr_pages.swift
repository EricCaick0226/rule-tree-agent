import Foundation
import Vision

struct OCRLine: Codable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct OCRPage: Codable {
    let image: String
    let page: Int
    let lines: [OCRLine]
}

func usage() {
    print("Usage: swift scripts/vision_ocr_pages.swift <image_dir_or_file> <output_dir>")
}

func pageNumber(_ url: URL) -> Int {
    let name = url.deletingPathExtension().lastPathComponent
    let pattern = #"page_(\d+)"#
    guard let regex = try? NSRegularExpression(pattern: pattern),
          let match = regex.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)),
          let range = Range(match.range(at: 1), in: name) else {
        return 1
    }
    return Int(name[range]) ?? 1
}

func imageFiles(from input: URL) -> [URL] {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: input.path, isDirectory: &isDirectory) else {
        return []
    }

    if !isDirectory.boolValue {
        return [input]
    }

    let files = (try? FileManager.default.contentsOfDirectory(
        at: input,
        includingPropertiesForKeys: nil
    )) ?? []

    return files
        .filter { ["png", "jpg", "jpeg", "tif", "tiff"].contains($0.pathExtension.lowercased()) }
        .sorted {
            let left = pageNumber($0)
            let right = pageNumber($1)
            if left != right { return left < right }
            return $0.lastPathComponent < $1.lastPathComponent
        }
}

func recognitionLanguages() -> [String] {
    let value = ProcessInfo.processInfo.environment["OCR_LANGUAGES"] ?? "zh-Hans,en-US"
    let languages = value
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    return languages.isEmpty ? ["zh-Hans", "en-US"] : languages
}

func recognizeText(in imageURL: URL) throws -> OCRPage {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = recognitionLanguages()
    request.minimumTextHeight = 0.006

    let handler = VNImageRequestHandler(url: imageURL, options: [:])
    try handler.perform([request])

    let observations = request.results ?? []
    let lines = observations.compactMap { observation -> OCRLine? in
        guard let candidate = observation.topCandidates(1).first else {
            return nil
        }

        let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty {
            return nil
        }

        let box = observation.boundingBox
        return OCRLine(
            text: text,
            confidence: candidate.confidence,
            x: box.minX,
            y: 1.0 - box.maxY,
            width: box.width,
            height: box.height
        )
    }

    return OCRPage(
        image: imageURL.lastPathComponent,
        page: pageNumber(imageURL),
        lines: lines.sorted {
            if abs($0.y - $1.y) > 0.005 {
                return $0.y < $1.y
            }
            return $0.x < $1.x
        }
    )
}

let args = CommandLine.arguments
guard args.count == 3 else {
    usage()
    exit(2)
}

let inputURL = URL(fileURLWithPath: args[1])
let outputURL = URL(fileURLWithPath: args[2])
do {
    try FileManager.default.createDirectory(
        at: outputURL,
        withIntermediateDirectories: true,
        attributes: nil
    )
} catch {
    fputs("Could not create OCR output directory: \(error)\n", stderr)
    exit(1)
}

let files = imageFiles(from: inputURL)
guard !files.isEmpty else {
    fputs("No image files found: \(inputURL.path)\n", stderr)
    exit(1)
}

var combinedBlocks: [String] = []
for file in files {
    let page: OCRPage
    do {
        page = try recognizeText(in: file)
    } catch {
        fputs("Vision OCR failed for \(file.path): \(error)\n", stderr)
        exit(1)
    }
    let text = page.lines.map { $0.text }.joined(separator: "\n")
    let baseName = file.deletingPathExtension().lastPathComponent

    let pageOutputURL = outputURL.appendingPathComponent(baseName + ".txt")
    do {
        try text.write(to: pageOutputURL, atomically: true, encoding: .utf8)
    } catch {
        fputs("Could not write OCR text output: \(error)\n", stderr)
        exit(1)
    }

    let pageJSONURL = outputURL.appendingPathComponent(baseName + ".json")
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    do {
        let jsonData = try encoder.encode(page)
        try jsonData.write(to: pageJSONURL, options: .atomic)
    } catch {
        fputs("Could not write OCR JSON output: \(error)\n", stderr)
        exit(1)
    }

    combinedBlocks.append("===== \(file.lastPathComponent) =====\n" + text)
    print("OCR \(file.lastPathComponent): \(text.count) chars")
}

let combined = combinedBlocks.joined(separator: "\n\n")
do {
    try combined.write(
        to: outputURL.appendingPathComponent("combined.txt"),
        atomically: true,
        encoding: .utf8
    )
} catch {
    fputs("Could not write combined OCR output: \(error)\n", stderr)
    exit(1)
}
