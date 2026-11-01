import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const csvPath = path.join(root, 'caption_references.csv');
const csv = await fs.readFile(csvPath, 'utf8');
const backup = await fs.readFile(path.join(here, 'original_caption_references.csv'), 'utf8');
if (csv !== backup) throw new Error('Source CSV changed after visual inspection; stop before overwrite');
const workbook = await Workbook.fromCSV(csv, { sheetName: 'References' });
const sheet = workbook.worksheets.getItem('References');

if (process.argv.includes('--discover')) {
  console.log(workbook.help('Workbook.toCSV', { include: 'index,examples,notes', maxChars: 3000 }).ndjson);
  const before = await workbook.render({ sheetName: 'References', range: 'A1:I5', scale: 1, format: 'png' });
  await fs.writeFile(path.join(here, 'before.png'), new Uint8Array(await before.arrayBuffer()));
} else {
  const captions = JSON.parse(await fs.readFile(path.join(here, 'captions.json'), 'utf8'));
  const teacher = 'openai-codex:model-id-unavailable';
  if (Object.keys(captions).length !== 188) throw new Error('Incomplete image annotation set');
  const immutable = sheet.getRange('A1:C189').values;
  const edits = Array.from({ length: 188 }, (_, index) => [
    captions[String(index + 1)], 'teacher', 'approved', 'ducsinh', teacher, '',
  ]);
  sheet.getRange('D2:I189').values = edits;
  workbook.recalculate();
  if (JSON.stringify(sheet.getRange('A1:C189').values) !== JSON.stringify(immutable)) {
    throw new Error('Immutable identifiers or drafts changed');
  }
  console.log((await workbook.inspect({ kind: 'table', range: 'References!D1:I4', include: 'values',
    tableMaxRows: 4, tableMaxCols: 6, maxChars: 3500 })).ndjson);
  const values = sheet.getRange('A1:I189').values.map(row => row.map(value => value ?? ''));
  await fs.writeFile(path.join(here, 'workbook_values.json'), JSON.stringify(values));
  // Formatting is for the inspection PNG only; CSV output has no workbook formatting.
  sheet.getRange('A1:I9').format.font = { name: 'Arial', size: 10 };
  sheet.getRange('A1:A9').format.columnWidthPx = 118;
  sheet.getRange('B1:B9').format.columnWidthPx = 235;
  sheet.getRange('C1:C9').format.columnWidthPx = 145;
  sheet.getRange('D1:D9').format.columnWidthPx = 360;
  sheet.getRange('E1:G9').format.columnWidthPx = 110;
  sheet.getRange('H1:H9').format.columnWidthPx = 270;
  sheet.getRange('I1:I9').format.columnWidthPx = 70;
  sheet.getRange('A1:I9').format.rowHeightPx = 36;
  sheet.getRange('A1:I9').format.wrapText = true;
  const after = await workbook.render({ sheetName: 'References', range: 'A1:I9', scale: 1, format: 'png' });
  await fs.writeFile(path.join(here, 'after.png'), new Uint8Array(await after.arrayBuffer()));
  console.log('Authored 188 reference rows; typed values saved for RFC-compatible CSV serialization.');
}
