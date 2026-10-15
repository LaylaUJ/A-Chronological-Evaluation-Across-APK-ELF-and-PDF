import fs from "node:fs/promises";
import path from "node:path";
import {
  FileBlob,
  SpreadsheetFile,
  Workbook,
} from "file:///C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const outputDir = process.argv[2] || "C:/workspace/paper_results";
const payloadPath = path.join(outputDir, "scripts", "table_payloads.json");
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const verificationDir = path.join(outputDir, "verification", "workbooks");
await fs.mkdir(verificationDir, { recursive: true });

const titleFill = "#17365D";
const headerFill = "#D9EAF7";
const accentFill = "#E2F0D9";
const borderColor = "#B7C9D6";

function prettyHeader(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function addDataSheet(workbook, name, title, rows, sourceText) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  const columnCount = rows[0].length;
  const lastColumn = columnName(columnCount);
  sheet.getRange(`A1:${lastColumn}1`).merge();
  sheet.getRange(`A1:${lastColumn}1`).format.fill = titleFill;
  sheet.getRange(`A1:${lastColumn}1`).format.font = {
    bold: true,
    color: "#FFFFFF",
    size: 14,
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 28;
  sheet.getRange(`A2:${lastColumn}2`).merge();
  sheet.getRange("A2").values = [[sourceText]];
  sheet.getRange(`A2:${lastColumn}2`).format.fill = accentFill;
  sheet.getRange(`A2:${lastColumn}2`).format.font = { italic: true, color: "#385723" };
  sheet.getRange(`A2:${lastColumn}2`).format.wrapText = true;
  const body = [rows[0].map(prettyHeader), ...rows.slice(1)];
  const endRow = body.length + 3;
  sheet.getRangeByIndexes(3, 0, body.length, columnCount).values = body;
  sheet.getRange(`A4:${lastColumn}4`).format.fill = headerFill;
  sheet.getRange(`A4:${lastColumn}4`).format.font = { bold: true, color: "#17365D" };
  sheet.getRange(`A4:${lastColumn}${endRow}`).format.borders = {
    preset: "all",
    style: "thin",
    color: borderColor,
  };
  sheet.getRange(`A4:${lastColumn}${endRow}`).format.verticalAlignment = "center";
  sheet.getRange(`A4:${lastColumn}${endRow}`).format.autofitColumns();
  sheet.getRange(`A4:${lastColumn}${endRow}`).format.autofitRows();
  sheet.freezePanes.freezeRows(4);
  // Keep prose columns readable and avoid extremely wide sheets.
  for (let col = 0; col < columnCount; col += 1) {
    const header = String(rows[0][col]);
    const columnRange = sheet.getRangeByIndexes(4, col, body.length - 1, 1);
    if ([
      "macro_f1", "ci_lower", "ci_upper", "mcc", "pr_auc", "roc_auc",
      "balanced_accuracy", "precision", "recall", "latency_ms", "model_size_mb",
      "jaccard", "spearman_rho", "p_value", "adjusted_p_value", "raw_p_value",
      "effect_tcfs_minus_baseline", "effect_a_minus_b", "effect_on_macro_f1",
      "macro_f1_mean", "macro_f1_sd", "macro_f1_ci_lower", "macro_f1_ci_upper",
      "pr_auc_mean", "mcc_mean", "jaccard_mean", "jaccard_sd", "jaccard_min",
      "jaccard_max", "spearman_mean", "spearman_sd", "spearman_min", "spearman_max",
    ].includes(header)) {
      columnRange.setNumberFormat("0.0000");
    }
    if (header === "feature_fraction") {
      columnRange.setNumberFormat("0.0%");
    }
    if (header.includes("analysis_unit") || header.includes("seed_or") || header.includes("correction")) {
      sheet.getRangeByIndexes(3, col, body.length, 1).format.columnWidth = 28;
      sheet.getRangeByIndexes(3, col, body.length, 1).format.wrapText = true;
    }
    if (header.includes("decision")) {
      sheet.getRangeByIndexes(3, col, body.length, 1).format.columnWidth = 18;
    }
    if (header.includes("method") || header === "selector" || header === "ablation") {
      sheet.getRangeByIndexes(3, col, body.length, 1).format.columnWidth = 22;
    }
  }
  sheet.getRange(`A4:${lastColumn}${endRow}`).format.autofitRows();
  return sheet;
}

function columnName(count) {
  let value = count;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

async function exportAndVerify(fileName, sheets) {
  const workbook = Workbook.create();
  for (const sheet of sheets) {
    addDataSheet(workbook, sheet.name, sheet.title, sheet.rows, sheet.source);
  }
  workbook.recalculate();
  const inspection = await workbook.inspect({
    kind: "workbook,sheet,formula",
    maxChars: 10000,
    tableMaxRows: 8,
    tableMaxCols: 12,
  });
  const inspectionText = inspection.ndjson || String(inspection);
  const formulaErrors = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"]
    .filter((token) => inspectionText.includes(token));
  if (formulaErrors.length) {
    throw new Error(`${fileName} contains formula errors: ${formulaErrors.join(", ")}`);
  }
  for (const sheet of sheets) {
    const preview = await workbook.render({
      sheetName: sheet.name,
      autoCrop: "all",
      scale: 1,
      format: "png",
    });
    const bytes = new Uint8Array(await preview.arrayBuffer());
    await fs.writeFile(
      path.join(verificationDir, `${fileName.replace(".xlsx", "")}__${sheet.name}.png`),
      bytes,
    );
  }
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  const savedPath = path.join(outputDir, fileName);
  await xlsx.save(savedPath);
  const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(savedPath));
  const reopenedInspection = await reopened.inspect({
    kind: "workbook,sheet,formula",
    maxChars: 10000,
    tableMaxRows: 8,
    tableMaxCols: 12,
  });
  const reopenedText = reopenedInspection.ndjson || String(reopenedInspection);
  const reopenedErrors = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"]
    .filter((token) => reopenedText.includes(token));
  if (reopenedErrors.length) {
    throw new Error(`${fileName} failed post-export verification: ${reopenedErrors.join(", ")}`);
  }
  await fs.writeFile(
    path.join(verificationDir, `${fileName}.inspection.txt`),
    `${inspectionText}\n\nPOST-EXPORT REOPEN\n${reopenedText}`,
    "utf8",
  );
}

await exportAndVerify("table_primary_results.xlsx", [
  {
    name: "Primary Results",
    title: "Future-test predictive performance and efficiency",
    rows: payload.primary_results,
    source: "Official sealed test weeks 53–64; frozen models refit on weeks 1–52; point metrics recalculated from saved sample-level predictions; 95% macro-F1 CIs use 2,000 paired class-stratified bootstrap resamples.",
  },
]);

await exportAndVerify("table_noninferiority.xlsx", [
  {
    name: "Noninferiority",
    title: "TCFS versus all-features macro-F1 non-inferiority",
    rows: payload.noninferiority,
    source: "Prespecified margin −0.01. Non-inferiority is declared only when the 95% CI lower bound exceeds −0.01.",
  },
  {
    name: "Secondary",
    title: "Confirmatory secondary comparisons",
    rows: payload.secondary_comparisons,
    source: "Paired 2,000-sample class-stratified bootstrap differences; Holm adjustment within each format across eight secondary metrics.",
  },
]);

await exportAndVerify("table_stability.xlsx", [
  {
    name: "Summary",
    title: "Temporal feature-set stability summary",
    rows: payload.stability_summary,
    source: "Six pairwise comparisons among four discovery windows; variability is the sample SD across window pairs.",
  },
  {
    name: "Jaccard Detail",
    title: "Pairwise Jaccard feature stability",
    rows: payload.jaccard_detail,
    source: "Selected-feature overlap at each frozen method budget.",
  },
  {
    name: "Spearman Detail",
    title: "Pairwise rank stability",
    rows: payload.spearman_detail,
    source: "Spearman correlation over the common ranked feature universe for each discovery-window pair.",
  },
]);

await exportAndVerify("table_ablation.xlsx", [
  {
    name: "Ablation",
    title: "Validation ablation and rho-sensitivity analysis",
    rows: payload.ablation,
    source: "Validation weeks 41–52 only; frozen three-seed mean-probability ensemble. Ablation CIs refer to effects versus full TCFS; sensitivity CIs refer to absolute macro-F1.",
  },
]);

