import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const inputName = process.argv[2] ?? 'audit_kualitas_tables.json';
const outputName = process.argv[3] ?? 'audit_kualitas_data.xlsx';
const payload = JSON.parse(await fs.readFile(path.join(root, 'outputs', inputName), 'utf8'));
const wb = Workbook.create();
const previews = path.join(root, 'outputs', process.argv[4] ?? '.audit_preview');
await fs.mkdir(previews, { recursive: true });
const checks = [];
for (const [name, records] of Object.entries(payload)) {
  const sheet = wb.worksheets.add(name);
  sheet.showGridLines = false;
  const columns = [...new Set(records.flatMap(r => Object.keys(r)))];
  const keys = columns.length ? columns : ['Keterangan'];
  const rows = records.length ? records.map(r => keys.map(k => {
    const v = r[k] ?? null;
    if (typeof v === 'string' && /^\d{4}-\d\d-\d\d(?:[ T]\d\d:\d\d:\d\d)?$/.test(v)) {
      // Excel serial: kalender/jam asli tanpa konversi zona waktu oleh renderer.
      return Date.parse(v.slice(0,10) + 'T' + (v.length > 10 ? v.slice(11,19) : '00:00:00') + 'Z')/86400000 + 25569;
    }
    return typeof v === 'string' && v.startsWith('=') ? "'" + v : v;
  })) : [['Tidak ada kejadian pada seluruh stasiun yang diaudit.']];
  const matrix = [keys, ...rows];
  sheet.getRangeByIndexes(0,0,matrix.length,keys.length).values = matrix;
  const used = sheet.getRangeByIndexes(0,0,matrix.length,keys.length);
  used.format.font.name = 'Arial';
  used.format.font.size = 10;
  used.format.rowHeight = 21;
  const header = sheet.getRangeByIndexes(0,0,1,keys.length);
  header.format.fill = '#253D56'; header.format.font.color = '#FFFFFF';
  header.format.font.bold = true; header.format.wrapText = true;
  header.format.rowHeight = 56;
  for (let i=0;i<keys.length;i++) {
    const col = sheet.getRangeByIndexes(0,i,matrix.length,1);
    const lengths = matrix.map(r => String(r[i] ?? '').length);
    const isDateColumn = records.some(r => typeof r[keys[i]] === 'string' && /^\d{4}-\d\d-\d\d(?:[ T]\d\d:\d\d:\d\d)?$/.test(r[keys[i]]));
    const width = Math.max(isDateColumn ? 170 : 100, Math.min(340, Math.max(...lengths.slice(0,1000))*7+16));
    col.format.columnWidthPx = width;
    if (lengths.slice(1).some(n => n > 48)) {
      sheet.getRangeByIndexes(1,i,rows.length,1).format.wrapText = true;
      sheet.getRangeByIndexes(1,0,rows.length,keys.length).format.rowHeight = 48;
    }
    if (keys[i].includes('persen')) sheet.getRangeByIndexes(1,i,rows.length,1).setNumberFormat('0.00');
    if (isDateColumn) sheet.getRangeByIndexes(1,i,rows.length,1).setNumberFormat('yyyy-mm-dd hh:mm:ss');
  }
  if (process.argv[2]) {
    for (let r=0;r<rows.length;r++) {
      const lines = Math.max(1,...rows[r].map(v => typeof v==='string' ? Math.ceil(v.length/45) : 1));
      sheet.getRangeByIndexes(r+1,0,1,keys.length).format.rowHeight = Math.max(24,(lines+1)*14);
    }
  }
  if (name === 'Definisi') {
    sheet.getRangeByIndexes(0,0,matrix.length,1).format.columnWidthPx=200;
    sheet.getRangeByIndexes(0,1,matrix.length,1).format.columnWidthPx=850;
    sheet.getRangeByIndexes(1,1,rows.length,1).format.wrapText=true;
    sheet.getRangeByIndexes(1,0,rows.length,2).format.rowHeight=65;
  }
  if (inputName === 'tabel_lstm_preprocessing.json' && name === 'Konfigurasi') {
    // Artifact Tool recognizes ISO UTC automatically; format the mixed-value cell explicitly.
    sheet.getRange('C2').setNumberFormat('yyyy-mm-dd hh:mm:ss');
  }
  sheet.freezePanes.freezeRows(1);
  if (keys[0] === 'File' && keys[1] === 'Stasiun') sheet.freezePanes.freezeColumns(2);
  else if (keys[0] === 'Stasiun') sheet.freezePanes.freezeColumns(1);
  checks.push({sheet:name, records:records.length, columns:keys.length});
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Ringkasan!A1:F4',tableMaxRows:4,tableMaxCols:6,maxChars:2200})).ndjson);
for (const c of checks) {
  const preview = await wb.render({sheetName:c.sheet,range:`A1:${c.columns>3?'D':c.columns>2?'C':c.columns>1?'B':'A'}${Math.min(Math.max(c.records,1)+1,5)}`,scale:1,format:'png'});
  await fs.writeFile(path.join(previews,c.sheet+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
if (inputName === 'pemeriksaan_lanjutan_tables.json') {
  const preview = await wb.render({sheetName:'Ringkasan',range:'L1:N3',scale:1,format:'png'});
  await fs.writeFile(path.join(previews,'Ringkasan_implikasi.png'),new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(root,'outputs',outputName));
await fs.writeFile(path.join(previews,'export_checks.json'),JSON.stringify(checks,null,2));
console.log('Ekspor selesai: '+checks.length+' sheet.');
