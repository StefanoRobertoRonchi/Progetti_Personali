/*
Traduzione SAS/IML di BayesianMA/BMA_SSVS.py
Contiene:
1) %Bayesian_MA_SSVS : stima SSVS con Gibbs sampler
2) %PiP             : Posterior Inclusion Probability
3) %TopModels       : modelli piu frequenti e metriche R2

Uso previsto:
- dataset input con variabile risposta y e regressori numerici
- richiede SAS/IML
*/

%macro Bayesian_MA_SSVS(
    data=,
    y=,
    x=,
    out_prefix=BMA,
    n=1000,
    burn_in=500,
    a=,
    b=,
    tau=,
    c=,
    p=0.5,
    identity=0,
    add_intercept=1,
    normalize_x=1,
    normalize_y=0,
    seed=42
);
proc iml;
    call randseed(&seed.);

    use &data.;
    read all var {&y.} into yVec;
    read all var {&x.} into X;
    close &data.;

    y = colvec(yVec);
    nObs = nrow(X);

    if &add_intercept.=1 then do;
        X = j(nObs,1,1) || X;
    end;

    nVar = ncol(X);

    muX = j(1,nVar,0);
    sigX = j(1,nVar,1);
    muY = mean(y);
    sigY = std(y);

    if &normalize_x.=1 then do;
        muX = X[:,];
        sigX = std(X);
        if &add_intercept.=1 then do;
            muX[1] = 0;
            sigX[1] = 1;
        end;
        X = (X - repeat(muX,nObs,1)) / repeat(sigX,nObs,1);
    end;

    if &normalize_y.=1 then do;
        y = (y - muY) / sigY;
    end;

    XtX = X`*X;
    Xty = X`*y;
    R0  = inv(XtX);

    BetaOLS = solve(XtX, Xty);
    sigmaOLS = sqrt(ssq(y - X*BetaOLS) / (nObs - nVar));
    varBetaOLS = (sigmaOLS##2)#R0;
    sigmaSq = sigmaOLS##2;

    if &identity.=1 then do;
        R = i(nVar);
    end;
    else do;
        sX = sqrt(vecdiag(R0));
        R  = R0 / (sX*sX`);
    end;

    aPrior = &a.;
    bPrior = &b.;
    tauVal = &tau.;
    cVal   = &c.;
    pIncl  = &p.;

    if aPrior=. | bPrior=. then do;
        aPrior = 6;
        bPrior = (aPrior - 1) * (sigmaOLS##2);
    end;
    if tauVal=. then tauVal = 20;
    if cVal=. then cVal = tauVal*10;

    tauSq = vecdiag(varBetaOLS) / (tauVal##2);

    nKeep = &n. - &burn_in.;
    gammaFinal = j(nKeep, nVar, 0);
    betaFinal  = j(nKeep, nVar, 0);
    sigFinal   = j(nKeep, 1, 0);

    gammas = j(nVar,1,1);

    do iter = 1 to &n.;
        aBetaVar = 1 + (cVal-1)#gammas;
        d = aBetaVar # sqrt(tauSq);
        D = diag(d);

        Vbeta = inv(inv(D*R*D) + XtX/sigmaSq);
        Bpost = Vbeta * Xty / sigmaSq;

        z = j(nVar,1,0);
        call randgen(z, "NORMAL");
        Betas = Bpost + root(Vbeta)*z;

        unif = j(nVar,1,0);
        call randgen(unif, "UNIFORM");

        sdSlab  = sqrt(tauSq)#cVal;
        sdSpike = sqrt(tauSq);

        /* log posterior inclusion probability */
        logNum = log(pIncl) + logpdf("NORMAL", Betas, 0, sdSlab);
        logAlt = log(1-pIncl) + logpdf("NORMAL", Betas, 0, sdSpike);
        mLog = choose(logNum>logAlt, logNum, logAlt);
        logDen = mLog + log(exp(logNum-mLog) + exp(logAlt-mLog));
        pGamma1 = exp(logNum - logDen);

        if &add_intercept.=1 then do;
            gammas[1] = 1;
            do j = 2 to nVar;
                gammas[j] = (unif[j] < pGamma1[j]);
            end;
        end;
        else gammas = (unif < pGamma1);

        rss = ssq(y - X*Betas);
        shape = aPrior + nObs/2;
        scale = 1/(bPrior + rss/2);
        gdraw = j(1,1,0);
        call randgen(gdraw, "GAMMA", shape, scale);
        sigmaSq = 1/gdraw;

        if iter > &burn_in. then do;
            idx = iter - &burn_in.;
            gammaFinal[idx,] = gammas`;
            betaFinal[idx,]  = Betas`;
            sigFinal[idx,1]  = sigmaSq;
        end;
    end;

    if &normalize_x.=1 & &normalize_y.=1 then do;
        betaFinal[,2:nVar] = betaFinal[,2:nVar] / repeat(sigX[2:nVar], nKeep, 1) * sigY;
        betaFinal[,1] = muY + betaFinal[,1]*sigY - betaFinal[,2:nVar]* (muX[2:nVar]`#(sigY/sigX[2:nVar])`);
        sigFinal = sigFinal # (sigY##2);
    end;
    else if &normalize_x.=1 then do;
        betaFinal[,2:nVar] = betaFinal[,2:nVar] / repeat(sigX[2:nVar], nKeep, 1);
        betaFinal[,1] = betaFinal[,1] - betaFinal[,2:nVar]* (muX[2:nVar]`/sigX[2:nVar]`);
    end;
    else if &normalize_y.=1 then do;
        betaFinal[,1] = betaFinal[,1]*sigY + muY;
        betaFinal[,2:nVar] = betaFinal[,2:nVar]*sigY;
        sigFinal = sigFinal # (sigY##2);
    end;

    modelNumber = t(1:nKeep);

    create &out_prefix._models var {modelNumber}; append; close &out_prefix._models;

    gammaNames = j(1,nVar,"gamma");
    do j=1 to nVar; gammaNames[j] = cats("gamma",j); end;
    create &out_prefix._gamma from gammaFinal[colname=gammaNames]; append from gammaFinal; close &out_prefix._gamma;

    betaNames = j(1,nVar,"beta");
    do j=1 to nVar; betaNames[j] = cats("beta",j); end;
    create &out_prefix._betas from betaFinal[colname=betaNames]; append from betaFinal; close &out_prefix._betas;

    create &out_prefix._sigma from sigFinal[colname={model_variance}]; append from sigFinal; close &out_prefix._sigma;

    addIntercept = &add_intercept.;
    create &out_prefix._meta var {addIntercept}; append; close &out_prefix._meta;
quit;
%mend;

%macro PiP(gamma_data=, out=PiP_out);
proc means data=&gamma_data. noprint;
    var gamma:;
    output out=&out.(drop=_TYPE_ _FREQ_) mean=;
run;
%mend;

%macro TopModels(data=, y=, x=, gamma_data=, beta_data=, out=TopModels_out, n_models=5);
/*
Combina catena gamma e beta, calcola R2 per draw e aggrega per combinazione di gamma.
*/
proc iml;
    use &data.;
    read all var {&y.} into y;
    read all var {&x.} into X;
    close &data.;

    use &gamma_data.; read all into G[colname=gNames]; close &gamma_data.;
    use &beta_data.;  read all into B[colname=bNames]; close &beta_data.;

    nDraw = nrow(B);
    if ncol(X)+1 = ncol(B) then X = j(nrow(X),1,1) || X;

    TSS = ssq(y - mean(y));
    R2 = j(nDraw,1,.);
    do i=1 to nDraw;
        b = B[i,]`;
        rss = ssq(y - X*b);
        R2[i] = 1 - rss/TSS;
    end;

    outM = G || R2;
    outNames = gNames || "R2";
    create __chain from outM[colname=outNames];
    append from outM;
    close __chain;
quit;

proc summary data=__chain nway;
    class gamma:;
    var R2;
    output out=__top(drop=_TYPE_ _FREQ_)
        mean=R2_mean
        median=R2_median
        p5=R2_q05
        p95=R2_q95
        n=count;
run;

proc sort data=__top; by descending count; run;

data &out.;
    set __top(obs=&n_models.);
run;

proc datasets library=work nolist;
    delete __chain __top;
quit;
%mend;

/* Esempio:
%Bayesian_MA_SSVS(data=mydata, y=y, x=x1 x2 x3 x4 x5 x6 x7, out_prefix=BMA, n=10000, burn_in=500, add_intercept=0);
%PiP(gamma_data=BMA_gamma, out=PIP_out);
*/
