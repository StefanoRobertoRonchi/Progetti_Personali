Attribute VB_Name = "BMA_SSVS_Excel"
Option Explicit

' ============================================================
' Bayesian Model Averaging - SSVS (Excel VBA version)
' Porting from Python script BMA_SSVS.py
' ============================================================
'
' Main macro for end users:
'   RunBayesianSSVSInteractive
'
' Workflow:
' 1) ask user to select data table (headers in first row)
' 2) ask target column (y)
' 3) ask regressors columns (x)
' 4) run Gibbs sampler and write results in dedicated sheets
'

Private Type BMAResult
    GammaFinal() As Double
    BetasFinal() As Double
    SigFinal() As Double
    AddedIntercept As Boolean
    RegNames() As String
    Y() As Double
    Xraw() As Double
End Type

Public Sub RunBayesianSSVSInteractive()
    On Error GoTo ErrHandler

    Dim dataRange As Range
    Dim targetInput As String, regInput As String
    Dim nIter As Long, burnIn As Long
    Dim normalizeX As Boolean, normalizeY As Boolean
    Dim addIntercept As Boolean, identityR As Boolean
    Dim tau As Double, cVal As Double, p As Double, a As Double, b As Double
    Dim seed As Long
    Dim y() As Double, X() As Double, regNames() As String
    Dim out As BMAResult

    MsgBox "BMA-SSVS VBA: seleziona la tabella dati (prima riga = header).", vbInformation

    Set dataRange = Application.InputBox( _
        Prompt:="Seleziona il range completo dei dati (inclusi header).", _
        Title:="BMA-SSVS | Range dati", Type:=8)

    targetInput = Trim$(InputBox( _
        Prompt:="Specifica la colonna target y (nome header o indice relativo, es: 3)", _
        Title:="BMA-SSVS | Target y"))
    If targetInput = "" Then Exit Sub

    regInput = Trim$(InputBox( _
        Prompt:="Specifica i regressori x separati da virgola (header o indici). Es: X1,X3,X7", _
        Title:="BMA-SSVS | Regressori x"))
    If regInput = "" Then Exit Sub

    nIter = CLng(GetDefault(InputBox("Numero iterazioni Gibbs (n)", "BMA-SSVS", "1000"), "1000"))
    burnIn = CLng(GetDefault(InputBox("Burn-in", "BMA-SSVS", "500"), "500"))
    If burnIn >= nIter Then Err.Raise 5, , "burn_in deve essere minore di n."

    addIntercept = AskYesNo("Aggiungere intercetta?", True)
    normalizeX = AskYesNo("Normalizzare X? (consigliato)", True)
    normalizeY = AskYesNo("Normalizzare y?", False)
    identityR = AskYesNo("Usare R=I invece di Corr((X'X)^-1)?", False)

    tau = CDbl(GetDefault(InputBox("tau (vuoto = 20)", "BMA-SSVS", "20"), "20"))
    cVal = CDbl(GetDefault(InputBox("c (vuoto = tau*10)", "BMA-SSVS", CStr(tau * 10#)), CStr(tau * 10#)))
    p = CDbl(GetDefault(InputBox("p prior inclusion (0-1)", "BMA-SSVS", "0.5"), "0.5"))
    a = CDbl(GetDefault(InputBox("a prior IG sigma^2 (vuoto=6)", "BMA-SSVS", "6"), "6"))
    b = CDbl(GetDefault(InputBox("b prior IG sigma^2 (vuoto auto da OLS)", "BMA-SSVS", ""), "-1"))
    seed = CLng(GetDefault(InputBox("Seed random", "BMA-SSVS", "42"), "42"))

    ParseXYFromRange dataRange, targetInput, regInput, y, X, regNames

    out = BayesianMASSVS(y, X, regNames, nIter, burnIn, a, b, tau, cVal, p, identityR, addIntercept, normalizeX, normalizeY, seed)

    WritePIP out, "BMA_PIP"
    WriteTopModels out, 5, "BMA_TopModels"
    WritePosteriorDraws out, "BMA_Draws"

    MsgBox "Completato. Risultati scritti nei fogli: BMA_PIP, BMA_TopModels, BMA_Draws", vbInformation
    Exit Sub

ErrHandler:
    MsgBox "Errore: " & Err.Description, vbCritical
End Sub

Private Function BayesianMASSVS(ByRef y0() As Double, ByRef X0() As Double, ByRef regNames0() As String, _
    ByVal nIter As Long, ByVal burnIn As Long, _
    ByVal a As Double, ByVal b As Double, ByVal tau As Double, ByVal cVal As Double, ByVal p As Double, _
    ByVal identityR As Boolean, ByVal addIntercept As Boolean, _
    ByVal normalizeX As Boolean, ByVal normalizeY As Boolean, ByVal seed As Long) As BMAResult

    Dim i As Long, j As Long, t As Long
    Dim nObs As Long, nVar As Long, storeN As Long
    Dim y() As Double, X() As Double, regNames() As String
    Dim muX() As Double, sdX() As Double, muY As Double, sdY As Double
    Dim Xt() As Double, XtX() As Double, Xty() As Double
    Dim R0() As Double, R() As Double
    Dim betaOLS() As Double, resid() As Double, sigmaOLS As Double, sigmaSq As Double
    Dim varBetaOLS() As Double, tauSq() As Double
    Dim gammas() As Double, betas() As Double
    Dim Vbeta() As Double, betaPost() As Double, cholV() As Double, z() As Double
    Dim aBetaVar() As Double, d() As Double, Dm() As Double
    Dim invPart() As Double, postK() As Double
    Dim unif As Double, sdSlab As Double, sdSpike As Double
    Dim logNum As Double, logDen As Double, pGamma1 As Double
    Dim rss As Double

    Randomize seed

    y = CopyVector(y0)
    X = CopyMatrix(X0)
    regNames = CopyStringVector(regNames0)

    nObs = UBound(y)
    nVar = UBound(X, 2)

    If addIntercept Then
        X = AddInterceptCol(X)
        regNames = AddInterceptName(regNames)
        nVar = nVar + 1
    End If

    If normalizeX Then
        ReDim muX(1 To nVar)
        ReDim sdX(1 To nVar)
        For j = 1 To nVar
            muX(j) = MeanCol(X, j)
            sdX(j) = StdCol(X, j)
            If addIntercept And j = 1 Then
                muX(j) = 0#
                sdX(j) = 1#
            ElseIf sdX(j) = 0# Then
                sdX(j) = 1#
            End If
            For i = 1 To nObs
                X(i, j) = (X(i, j) - muX(j)) / sdX(j)
            Next i
        Next j
    End If

    If normalizeY Then
        muY = MeanVec(y)
        sdY = StdVec(y)
        If sdY = 0# Then sdY = 1#
        For i = 1 To nObs
            y(i) = (y(i) - muY) / sdY
        Next i
    End If

    Xt = TransposeMat(X)
    XtX = MatMul(Xt, X)
    Xty = MatVecMul(Xt, y)
    R0 = MatInv(XtX)

    betaOLS = SolveLinear(XtX, Xty)
    resid = VecSub(y, MatVecMul(X, betaOLS))
    sigmaOLS = Sqr(VecDot(resid, resid) / (nObs - nVar))
    sigmaSq = sigmaOLS * sigmaOLS

    varBetaOLS = MatScale(R0, sigmaSq)

    If identityR Then
        R = IdentityMat(nVar)
    Else
        R = CorFromCov(R0)
    End If

    If a <= 0# Then a = 6#
    If b <= 0# Then b = (a - 1#) * sigmaSq
    If tau <= 0# Then tau = 20#
    If cVal <= 0# Then cVal = tau * 10#
    If p <= 0# Or p >= 1# Then p = 0.5

    ReDim tauSq(1 To nVar)
    For j = 1 To nVar
        tauSq(j) = varBetaOLS(j, j) / (tau * tau)
        If tauSq(j) <= 0# Then tauSq(j) = 1E-8
    Next j

    ReDim gammas(1 To nVar)
    For j = 1 To nVar
        gammas(j) = 1#
    Next j

    storeN = nIter - burnIn
    Dim gammaFinal() As Double, betasFinal() As Double, sigFinal() As Double
    ReDim gammaFinal(1 To storeN, 1 To nVar)
    ReDim betasFinal(1 To storeN, 1 To nVar)
    ReDim sigFinal(1 To storeN)

    For t = 1 To nIter
        ReDim aBetaVar(1 To nVar)
        ReDim d(1 To nVar)
        For j = 1 To nVar
            aBetaVar(j) = 1# + (cVal - 1#) * gammas(j)
            d(j) = aBetaVar(j) * Sqr(tauSq(j))
        Next j

        Dm = DiagMat(d)
        invPart = MatInv(MatMul(MatMul(Dm, R), Dm))
        postK = MatAdd(invPart, MatScale(XtX, 1# / sigmaSq))
        Vbeta = MatInv(postK)
        betaPost = MatVecScale(MatVecMul(Vbeta, Xty), 1# / sigmaSq)

        cholV = Cholesky(Vbeta)
        ReDim z(1 To nVar)
        For j = 1 To nVar
            z(j) = RandNormal()
        Next j
        betas = VecAdd(betaPost, MatVecMul(cholV, z))

        For j = 1 To nVar
            sdSlab = Sqr(tauSq(j)) * cVal
            sdSpike = Sqr(tauSq(j))
            logNum = Log(p) + LogNormPdf(betas(j), 0#, sdSlab)
            logDen = LogSumExp(logNum, Log(1# - p) + LogNormPdf(betas(j), 0#, sdSpike))
            pGamma1 = Exp(logNum - logDen)

            If addIntercept And j = 1 Then
                gammas(j) = 1#
            Else
                unif = Rnd()
                If unif < pGamma1 Then
                    gammas(j) = 1#
                Else
                    gammas(j) = 0#
                End If
            End If
        Next j

        resid = VecSub(y, MatVecMul(X, betas))
        rss = VecDot(resid, resid)
        sigmaSq = 1# / RandGammaMarsaglia(a + nObs / 2#, 1# / (b + rss / 2#))

        If t > burnIn Then
            i = t - burnIn
            For j = 1 To nVar
                gammaFinal(i, j) = gammas(j)
                betasFinal(i, j) = betas(j)
            Next j
            sigFinal(i) = sigmaSq
        End If
    Next t

    If normalizeX Or normalizeY Then
        DenormalizeBetas betasFinal, sigFinal, muX, sdX, muY, sdY, normalizeX, normalizeY, addIntercept
    End If

    Dim result As BMAResult
    result.GammaFinal = gammaFinal
    result.BetasFinal = betasFinal
    result.SigFinal = sigFinal
    result.AddedIntercept = addIntercept
    result.RegNames = regNames
    result.Y = y0
    result.Xraw = X0

    BayesianMASSVS = result
End Function

Private Sub DenormalizeBetas(ByRef betasFinal() As Double, ByRef sigFinal() As Double, _
    ByRef muX() As Double, ByRef sdX() As Double, ByVal muY As Double, ByVal sdY As Double, _
    ByVal normalizeX As Boolean, ByVal normalizeY As Boolean, ByVal addIntercept As Boolean)

    Dim i As Long, j As Long, nDraw As Long, nVar As Long
    Dim correction As Double

    nDraw = UBound(betasFinal, 1)
    nVar = UBound(betasFinal, 2)

    For i = 1 To nDraw
        correction = 0#
        If normalizeX Then
            For j = IIf(addIntercept, 2, 1) To nVar
                correction = correction + betasFinal(i, j) * (muX(j) / sdX(j))
                betasFinal(i, j) = betasFinal(i, j) / sdX(j)
            Next j
            betasFinal(i, 1) = betasFinal(i, 1) - correction
        End If

        If normalizeY Then
            betasFinal(i, 1) = betasFinal(i, 1) * sdY + muY
            For j = IIf(addIntercept, 2, 1) To nVar
                betasFinal(i, j) = betasFinal(i, j) * sdY
            Next j
            sigFinal(i) = sigFinal(i) * (sdY * sdY)
        End If
    Next i
End Sub

Private Sub WritePIP(ByRef out As BMAResult, ByVal sheetName As String)
    Dim ws As Worksheet, i As Long, j As Long, nDraw As Long, nVar As Long
    Set ws = PrepareSheet(sheetName)

    nDraw = UBound(out.GammaFinal, 1)
    nVar = UBound(out.GammaFinal, 2)

    ws.Range("A1").Value = "Regressor"
    ws.Range("B1").Value = "PIP"

    For j = 1 To nVar
        ws.Cells(j + 1, 1).Value = out.RegNames(j)
        ws.Cells(j + 1, 2).Value = MeanArrayCol(out.GammaFinal, j)
    Next j

    ws.Columns("A:B").AutoFit
End Sub

Private Sub WriteTopModels(ByRef out As BMAResult, ByVal nModels As Long, ByVal sheetName As String)
    Dim ws As Worksheet
    Dim dict As Object
    Dim i As Long, j As Long, k As Long, nDraw As Long, nVar As Long
    Dim key As String, r2 As Double
    Dim y() As Double, X() As Double, beta() As Double, gamma() As Double
    Dim tss As Double, rss As Double
    Dim modelKeys() As String, counts() As Long

    Set ws = PrepareSheet(sheetName)
    Set dict = CreateObject("Scripting.Dictionary")

    y = CopyVector(out.Y)
    X = CopyMatrix(out.Xraw)
    If out.AddedIntercept Then X = AddInterceptCol(X)

    nDraw = UBound(out.GammaFinal, 1)
    nVar = UBound(out.GammaFinal, 2)

    tss = TSS(y)

    For i = 1 To nDraw
        ReDim beta(1 To nVar)
        ReDim gamma(1 To nVar)
        For j = 1 To nVar
            beta(j) = out.BetasFinal(i, j)
            gamma(j) = out.GammaFinal(i, j)
        Next j

        rss = VecDot(VecSub(y, MatVecMul(X, beta)), VecSub(y, MatVecMul(X, beta)))
        r2 = 1# - rss / tss
        key = GammaKey(gamma)

        If Not dict.Exists(key) Then
            dict.Add key, Array(1&, r2, r2, r2, r2)
        Else
            Dim arr As Variant
            arr = dict(key)
            arr(0) = CLng(arr(0)) + 1
            arr(1) = CDbl(arr(1)) + r2
            arr(2) = WorksheetFunction.Median(Array(CDbl(arr(2)), r2))
            If r2 < CDbl(arr(3)) Then arr(3) = r2
            If r2 > CDbl(arr(4)) Then arr(4) = r2
            dict(key) = arr
        End If
    Next i

    ReDim modelKeys(1 To dict.Count)
    ReDim counts(1 To dict.Count)
    k = 0
    Dim kk As Variant
    For Each kk In dict.Keys
        k = k + 1
        modelKeys(k) = CStr(kk)
        counts(k) = CLng(dict(kk)(0))
    Next kk

    SortModelsByCount modelKeys, counts

    ws.Cells(1, 1).Value = "Model"
    For j = 1 To nVar
        ws.Cells(1, j + 1).Value = out.RegNames(j)
    Next j
    ws.Cells(1, nVar + 2).Value = "R2_mean"
    ws.Cells(1, nVar + 3).Value = "R2_median_proxy"
    ws.Cells(1, nVar + 4).Value = "R2_min_proxy"
    ws.Cells(1, nVar + 5).Value = "R2_max_proxy"
    ws.Cells(1, nVar + 6).Value = "count"

    Dim topN As Long: topN = WorksheetFunction.Min(nModels, dict.Count)
    For i = 1 To topN
        key = modelKeys(i)
        ws.Cells(i + 1, 1).Value = "Model_" & i
        For j = 1 To nVar
            ws.Cells(i + 1, j + 1).Value = Mid$(key, j, 1)
        Next j

        Dim m As Variant
        m = dict(key)
        ws.Cells(i + 1, nVar + 2).Value = CDbl(m(1)) / CLng(m(0))
        ws.Cells(i + 1, nVar + 3).Value = CDbl(m(2))
        ws.Cells(i + 1, nVar + 4).Value = CDbl(m(3))
        ws.Cells(i + 1, nVar + 5).Value = CDbl(m(4))
        ws.Cells(i + 1, nVar + 6).Value = CLng(m(0))
    Next i

    ws.Columns.AutoFit
End Sub

Private Sub WritePosteriorDraws(ByRef out As BMAResult, ByVal sheetName As String)
    Dim ws As Worksheet, i As Long, j As Long, nDraw As Long, nVar As Long
    Set ws = PrepareSheet(sheetName)

    nDraw = UBound(out.BetasFinal, 1)
    nVar = UBound(out.BetasFinal, 2)

    ws.Cells(1, 1).Value = "Draw"
    For j = 1 To nVar
        ws.Cells(1, j + 1).Value = "beta_" & out.RegNames(j)
        ws.Cells(1, nVar + 1 + j).Value = "gamma_" & out.RegNames(j)
    Next j
    ws.Cells(1, 2 * nVar + 2).Value = "sigma2"

    For i = 1 To nDraw
        ws.Cells(i + 1, 1).Value = i
        For j = 1 To nVar
            ws.Cells(i + 1, 1 + j).Value = out.BetasFinal(i, j)
            ws.Cells(i + 1, 1 + nVar + j).Value = out.GammaFinal(i, j)
        Next j
        ws.Cells(i + 1, 2 * nVar + 2).Value = out.SigFinal(i)
    Next i

    ws.Columns.AutoFit
End Sub

Private Sub ParseXYFromRange(ByVal dataRange As Range, ByVal targetInput As String, ByVal regInput As String, _
    ByRef y() As Double, ByRef X() As Double, ByRef regNames() As String)

    Dim data As Variant
    Dim nRows As Long, nCols As Long
    Dim tCol As Long, regCols() As Long
    Dim i As Long, j As Long
    Dim regTokens() As String

    data = dataRange.Value2
    nRows = UBound(data, 1)
    nCols = UBound(data, 2)

    tCol = ResolveColumn(data, nCols, targetInput)

    regTokens = Split(regInput, ",")
    ReDim regCols(1 To UBound(regTokens) - LBound(regTokens) + 1)
    For i = LBound(regTokens) To UBound(regTokens)
        regCols(i - LBound(regTokens) + 1) = ResolveColumn(data, nCols, Trim$(regTokens(i)))
    Next i

    ReDim y(1 To nRows - 1)
    ReDim X(1 To nRows - 1, 1 To UBound(regCols))
    ReDim regNames(1 To UBound(regCols))

    For j = 1 To UBound(regCols)
        regNames(j) = CStr(data(1, regCols(j)))
    Next j

    For i = 2 To nRows
        y(i - 1) = CDbl(data(i, tCol))
        For j = 1 To UBound(regCols)
            X(i - 1, j) = CDbl(data(i, regCols(j)))
        Next j
    Next i
End Sub

Private Function ResolveColumn(ByRef data As Variant, ByVal nCols As Long, ByVal token As String) As Long
    Dim j As Long
    If IsNumeric(token) Then
        ResolveColumn = CLng(token)
        If ResolveColumn < 1 Or ResolveColumn > nCols Then Err.Raise 5, , "Indice colonna fuori range: " & token
        Exit Function
    End If

    For j = 1 To nCols
        If StrComp(CStr(data(1, j)), token, vbTextCompare) = 0 Then
            ResolveColumn = j
            Exit Function
        End If
    Next j

    Err.Raise 5, , "Colonna non trovata: " & token
End Function

Private Function AskYesNo(ByVal prompt As String, ByVal defaultYes As Boolean) As Boolean
    Dim d As VbMsgBoxResult
    d = MsgBox(prompt, vbYesNo + IIf(defaultYes, vbDefaultButton1, vbDefaultButton2) + vbQuestion)
    AskYesNo = (d = vbYes)
End Function

Private Function GetDefault(ByVal txt As String, ByVal defaultValue As String) As String
    If Trim$(txt) = "" Then
        GetDefault = defaultValue
    Else
        GetDefault = txt
    End If
End Function

Private Function PrepareSheet(ByVal sheetName As String) As Worksheet
    On Error Resume Next
    Set PrepareSheet = ThisWorkbook.Worksheets(sheetName)
    On Error GoTo 0

    If PrepareSheet Is Nothing Then
        Set PrepareSheet = ThisWorkbook.Worksheets.Add(After:=ThisWorkbook.Worksheets(ThisWorkbook.Worksheets.Count))
        PrepareSheet.Name = sheetName
    Else
        PrepareSheet.Cells.Clear
    End If
End Function

' ---------- Math helpers ----------
Private Function MeanVec(ByRef v() As Double) As Double
    Dim i As Long, s As Double
    For i = LBound(v) To UBound(v)
        s = s + v(i)
    Next i
    MeanVec = s / (UBound(v) - LBound(v) + 1)
End Function

Private Function StdVec(ByRef v() As Double) As Double
    Dim i As Long, m As Double, s As Double, n As Long
    n = UBound(v) - LBound(v) + 1
    If n <= 1 Then StdVec = 0#: Exit Function
    m = MeanVec(v)
    For i = LBound(v) To UBound(v)
        s = s + (v(i) - m) ^ 2
    Next i
    StdVec = Sqr(s / (n - 1))
End Function

Private Function MeanCol(ByRef M() As Double, ByVal col As Long) As Double
    Dim i As Long, s As Double
    For i = LBound(M, 1) To UBound(M, 1)
        s = s + M(i, col)
    Next i
    MeanCol = s / (UBound(M, 1) - LBound(M, 1) + 1)
End Function

Private Function StdCol(ByRef M() As Double, ByVal col As Long) As Double
    Dim i As Long, m As Double, s As Double, n As Long
    n = UBound(M, 1) - LBound(M, 1) + 1
    If n <= 1 Then StdCol = 0#: Exit Function
    m = MeanCol(M, col)
    For i = LBound(M, 1) To UBound(M, 1)
        s = s + (M(i, col) - m) ^ 2
    Next i
    StdCol = Sqr(s / (n - 1))
End Function

Private Function TransposeMat(ByRef A() As Double) As Double()
    Dim i As Long, j As Long
    Dim R() As Double
    ReDim R(1 To UBound(A, 2), 1 To UBound(A, 1))
    For i = 1 To UBound(A, 1)
        For j = 1 To UBound(A, 2)
            R(j, i) = A(i, j)
        Next j
    Next i
    TransposeMat = R
End Function

Private Function MatMul(ByRef A() As Double, ByRef B() As Double) As Double()
    Dim i As Long, j As Long, k As Long
    Dim rA As Long, cA As Long, cB As Long
    Dim R() As Double

    rA = UBound(A, 1): cA = UBound(A, 2): cB = UBound(B, 2)
    ReDim R(1 To rA, 1 To cB)

    For i = 1 To rA
        For j = 1 To cB
            For k = 1 To cA
                R(i, j) = R(i, j) + A(i, k) * B(k, j)
            Next k
        Next j
    Next i
    MatMul = R
End Function

Private Function MatVecMul(ByRef A() As Double, ByRef v() As Double) As Double()
    Dim i As Long, j As Long
    Dim R() As Double
    ReDim R(1 To UBound(A, 1))
    For i = 1 To UBound(A, 1)
        For j = 1 To UBound(A, 2)
            R(i) = R(i) + A(i, j) * v(j)
        Next j
    Next i
    MatVecMul = R
End Function

Private Function MatScale(ByRef A() As Double, ByVal s As Double) As Double()
    Dim i As Long, j As Long
    Dim R() As Double
    ReDim R(1 To UBound(A, 1), 1 To UBound(A, 2))
    For i = 1 To UBound(A, 1)
        For j = 1 To UBound(A, 2)
            R(i, j) = A(i, j) * s
        Next j
    Next i
    MatScale = R
End Function

Private Function MatAdd(ByRef A() As Double, ByRef B() As Double) As Double()
    Dim i As Long, j As Long
    Dim R() As Double
    ReDim R(1 To UBound(A, 1), 1 To UBound(A, 2))
    For i = 1 To UBound(A, 1)
        For j = 1 To UBound(A, 2)
            R(i, j) = A(i, j) + B(i, j)
        Next j
    Next i
    MatAdd = R
End Function

Private Function MatInv(ByRef A() As Double) As Double()
    Dim v As Variant, inv As Variant
    v = MatrixToVariant(A)
    inv = WorksheetFunction.MInverse(v)
    MatInv = VariantToMatrix(inv)
End Function

Private Function SolveLinear(ByRef A() As Double, ByRef b() As Double) As Double()
    SolveLinear = MatVecMul(MatInv(A), b)
End Function

Private Function IdentityMat(ByVal n As Long) As Double()
    Dim i As Long, R() As Double
    ReDim R(1 To n, 1 To n)
    For i = 1 To n
        R(i, i) = 1#
    Next i
    IdentityMat = R
End Function

Private Function DiagMat(ByRef d() As Double) As Double()
    Dim i As Long, n As Long, R() As Double
    n = UBound(d)
    ReDim R(1 To n, 1 To n)
    For i = 1 To n
        R(i, i) = d(i)
    Next i
    DiagMat = R
End Function

Private Function CorFromCov(ByRef C() As Double) As Double()
    Dim i As Long, j As Long, n As Long
    Dim R() As Double, s() As Double
    n = UBound(C, 1)
    ReDim R(1 To n, 1 To n)
    ReDim s(1 To n)
    For i = 1 To n
        s(i) = Sqr(C(i, i))
        If s(i) = 0# Then s(i) = 1#
    Next i
    For i = 1 To n
        For j = 1 To n
            R(i, j) = C(i, j) / (s(i) * s(j))
        Next j
    Next i
    CorFromCov = R
End Function

Private Function Cholesky(ByRef A() As Double) As Double()
    Dim i As Long, j As Long, k As Long, n As Long
    Dim L() As Double, sum As Double
    n = UBound(A, 1)
    ReDim L(1 To n, 1 To n)

    For i = 1 To n
        For j = 1 To i
            sum = 0#
            For k = 1 To j - 1
                sum = sum + L(i, k) * L(j, k)
            Next k
            If i = j Then
                L(i, j) = Sqr(Application.Max(A(i, i) - sum, 1E-12))
            Else
                L(i, j) = (A(i, j) - sum) / L(j, j)
            End If
        Next j
    Next i
    Cholesky = L
End Function

Private Function VecSub(ByRef a() As Double, ByRef b() As Double) As Double()
    Dim i As Long, R() As Double
    ReDim R(1 To UBound(a))
    For i = 1 To UBound(a)
        R(i) = a(i) - b(i)
    Next i
    VecSub = R
End Function

Private Function VecAdd(ByRef a() As Double, ByRef b() As Double) As Double()
    Dim i As Long, R() As Double
    ReDim R(1 To UBound(a))
    For i = 1 To UBound(a)
        R(i) = a(i) + b(i)
    Next i
    VecAdd = R
End Function

Private Function VecDot(ByRef a() As Double, ByRef b() As Double) As Double
    Dim i As Long, s As Double
    For i = 1 To UBound(a)
        s = s + a(i) * b(i)
    Next i
    VecDot = s
End Function

Private Function MatVecScale(ByRef v() As Double, ByVal s As Double) As Double()
    Dim i As Long, R() As Double
    ReDim R(1 To UBound(v))
    For i = 1 To UBound(v)
        R(i) = v(i) * s
    Next i
    MatVecScale = R
End Function

Private Function CopyVector(ByRef v() As Double) As Double()
    Dim i As Long, r() As Double
    ReDim r(1 To UBound(v))
    For i = 1 To UBound(v)
        r(i) = v(i)
    Next i
    CopyVector = r
End Function

Private Function CopyStringVector(ByRef v() As String) As String()
    Dim i As Long, r() As String
    ReDim r(1 To UBound(v))
    For i = 1 To UBound(v)
        r(i) = v(i)
    Next i
    CopyStringVector = r
End Function

Private Function CopyMatrix(ByRef A() As Double) As Double()
    Dim i As Long, j As Long, R() As Double
    ReDim R(1 To UBound(A, 1), 1 To UBound(A, 2))
    For i = 1 To UBound(A, 1)
        For j = 1 To UBound(A, 2)
            R(i, j) = A(i, j)
        Next j
    Next i
    CopyMatrix = R
End Function

Private Function AddInterceptCol(ByRef X() As Double) As Double()
    Dim i As Long, j As Long
    Dim nObs As Long, nVar As Long
    Dim R() As Double

    nObs = UBound(X, 1)
    nVar = UBound(X, 2)
    ReDim R(1 To nObs, 1 To nVar + 1)

    For i = 1 To nObs
        R(i, 1) = 1#
        For j = 1 To nVar
            R(i, j + 1) = X(i, j)
        Next j
    Next i
    AddInterceptCol = R
End Function

Private Function AddInterceptName(ByRef names() As String) As String()
    Dim j As Long, R() As String
    ReDim R(1 To UBound(names) + 1)
    R(1) = "Intercept"
    For j = 1 To UBound(names)
        R(j + 1) = names(j)
    Next j
    AddInterceptName = R
End Function

Private Function MatrixToVariant(ByRef A() As Double) As Variant
    Dim i As Long, j As Long, V() As Variant
    ReDim V(1 To UBound(A, 1), 1 To UBound(A, 2))
    For i = 1 To UBound(A, 1)
        For j = 1 To UBound(A, 2)
            V(i, j) = A(i, j)
        Next j
    Next i
    MatrixToVariant = V
End Function

Private Function VariantToMatrix(ByRef V As Variant) As Double()
    Dim i As Long, j As Long, A() As Double
    ReDim A(1 To UBound(V, 1), 1 To UBound(V, 2))
    For i = 1 To UBound(V, 1)
        For j = 1 To UBound(V, 2)
            A(i, j) = CDbl(V(i, j))
        Next j
    Next i
    VariantToMatrix = A
End Function

Private Function RandNormal() As Double
    Dim u1 As Double, u2 As Double
    u1 = Rnd: If u1 <= 0# Then u1 = 1E-12
    u2 = Rnd
    RandNormal = Sqr(-2# * Log(u1)) * Cos(2# * WorksheetFunction.Pi() * u2)
End Function

Private Function RandGammaMarsaglia(ByVal shape As Double, ByVal scale As Double) As Double
    Dim d As Double, c As Double
    Dim x As Double, v As Double, u As Double

    If shape < 1# Then
        RandGammaMarsaglia = RandGammaMarsaglia(shape + 1#, scale) * (Rnd() ^ (1# / shape))
        Exit Function
    End If

    d = shape - 1# / 3#
    c = 1# / Sqr(9# * d)

    Do
        Do
            x = RandNormal()
            v = 1# + c * x
        Loop While v <= 0#
        v = v ^ 3
        u = Rnd

        If u < 1# - 0.0331 * x ^ 4 Then Exit Do
        If Log(u) < 0.5 * x * x + d * (1# - v + Log(v)) Then Exit Do
    Loop

    RandGammaMarsaglia = d * v * scale
End Function

Private Function LogNormPdf(ByVal x As Double, ByVal mu As Double, ByVal sigma As Double) As Double
    If sigma <= 0# Then sigma = 1E-8
    LogNormPdf = -0.5 * Log(2# * WorksheetFunction.Pi()) - Log(sigma) - ((x - mu) ^ 2) / (2# * sigma * sigma)
End Function

Private Function LogSumExp(ByVal a As Double, ByVal b As Double) As Double
    Dim m As Double
    m = IIf(a > b, a, b)
    LogSumExp = m + Log(Exp(a - m) + Exp(b - m))
End Function

Private Function MeanArrayCol(ByRef A() As Double, ByVal col As Long) As Double
    Dim i As Long, s As Double
    For i = 1 To UBound(A, 1)
        s = s + A(i, col)
    Next i
    MeanArrayCol = s / UBound(A, 1)
End Function

Private Function TSS(ByRef y() As Double) As Double
    Dim i As Long, m As Double, s As Double
    m = MeanVec(y)
    For i = 1 To UBound(y)
        s = s + (y(i) - m) ^ 2
    Next i
    TSS = s
End Function

Private Function GammaKey(ByRef gamma() As Double) As String
    Dim i As Long, s As String
    s = ""
    For i = 1 To UBound(gamma)
        s = s & CStr(CLng(gamma(i)))
    Next i
    GammaKey = s
End Function

Private Sub SortModelsByCount(ByRef keys() As String, ByRef counts() As Long)
    Dim i As Long, j As Long
    Dim ck As Long, kk As String
    For i = LBound(counts) To UBound(counts) - 1
        For j = i + 1 To UBound(counts)
            If counts(j) > counts(i) Then
                ck = counts(i): counts(i) = counts(j): counts(j) = ck
                kk = keys(i): keys(i) = keys(j): keys(j) = kk
            End If
        Next j
    Next i
End Sub
