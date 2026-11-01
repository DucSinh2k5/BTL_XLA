param(
    [Parameter(Mandatory=$true)][string]$Docx,
    [Parameter(Mandatory=$true)][string]$Pdf
)

$ErrorActionPreference = 'Stop'
$documentPath = (Resolve-Path -LiteralPath $Docx).Path
$pdfPath = [System.IO.Path]::GetFullPath($Pdf)
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($documentPath, $false, $false, $false)
    $document.Repaginate()
    $document.Fields.Update() | Out-Null
    foreach ($toc in $document.TablesOfContents) {
        $toc.Update()
        $toc.Range.Font.Name = 'Times New Roman'
        $toc.Range.Font.Size = 11
        $toc.Range.ParagraphFormat.SpaceBefore = 0
        $toc.Range.ParagraphFormat.SpaceAfter = 2
        $toc.Range.ParagraphFormat.LineSpacingRule = 0
    }
    $document.Repaginate()
    foreach ($toc in $document.TablesOfContents) { $toc.UpdatePageNumbers() }
    $document.Save()
    $document.ExportAsFixedFormat($pdfPath, 17)
    Write-Output ('Pages: ' + $document.ComputeStatistics(2))
    Write-Output ('PDF: ' + $pdfPath)
} finally {
    if ($null -ne $document) {
        $document.Close(0)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($null -ne $word) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
}
