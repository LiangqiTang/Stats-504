#!/usr/bin/env python
# coding: utf-8

# # iNaturalist data
# 
# [iNaturalist](https://inaturalist.org) is a platform that collects observations of living organisms, annotated with their locations and taxonomic identifiers.
# 
# One line of research that can be addressed using the iNaturalist data is to characterize temporal trends in the locations at which plants of a particular species are observed.  We will be considering data for plants here. The individual plants can be assumed to have fixed locations, but the range of a species can change over time.  Such range changes could be due to changes in environmental conditions (e.g. climate), or to changes in the behavior of the human observers.
# 
# This notebook illustrates some methods of [large scale inference](https://efron.ckirby.su.domains/other/2010LSIexcerpt.pdf) that can be used to identify systematic changes in species locations over time. 

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import cartopy.crs as ccrs
from scipy.stats.distributions import norm, chi2
from scipy.stats.distributions import t as tdist
from statsmodels.stats.multitest import local_fdr
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt


# Below we select a [class](https://en.wikipedia.org/wiki/Class_(biology)) of species to analyze.  A class is a taxonomic grouping that includes many species.  [Pinopsida](https://en.wikipedia.org/wiki/Conifer) are conifers and [polypodiopsida](https://en.wikipedia.org/wiki/Fern) are ferns.

# In[2]:


#pclass = "Pinopsida"
pclass = "Polypodiopsida"


# The data are located at this path:

# In[3]:


pa = Path("/Users/liangqi/Library/Mobile Documents/com~apple~CloudDocs/Stats-504/inaturalist")
fn = pa / ("Plantae_%s.csv.gz" % pclass)


# Below we load the data and show what the data file looks like.

# In[4]:


df = pd.read_csv(fn, parse_dates=["eventDate"])
print(df.shape)
df.head()


# The dates are mostly in the past 8 years, but a few dates are much older than that.  We will analyze observations made since 2015.

# In[5]:


df = df.query("eventDate >= 20150101")


# Next we construct a time variable that starts on January 1, 2015 and counts in 1000's of days from that origin.  We will be interested in the evidence that for specific species, the average latitude changes linearly as a function of this 'day' variable. This would suggest that the species range is trending away from the equator (either north or south).  A simplistic climate change framing would posit that plants in the northern hemisphere would tend to move northward and plants in the southern hemisphere would tend to move southward.

# In[6]:


df["day"] = (df["eventDate"] - pd.to_datetime("2015-01-01")).dt.days
df["day"] /= 1000


# This is the total number of observations (the number of plant occurrences in the dataset for the selected class of species):

# In[9]:


df.head()


# In[7]:


df.shape


# The next cell calculates the number of distinct species.

# In[10]:


df["species"].value_counts().size


# Below we make a histogram of the number of observations per species.

# In[11]:


nn = df.groupby("species").size()
plt.hist(np.log10(nn))
plt.xlabel("log10 number of observations")
plt.ylabel("Frequency")


# A quantile plot is usually more informative than a histogram.

# In[13]:


plt.clf()
plt.grid(True)
plt.plot(np.sort(np.log10(nn)))
plt.xlabel("Number of species")
plt.ylabel("log10 number of observations")


# The map below shows the distribution of the occurrences (the locations where a plant in the selected class was observed).

# In[14]:


plt.clf()
plt.figure(figsize=(9, 7.25))
ax = plt.axes([0.05, 0.05, 0.84, 0.88], projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines()
ax.set_extent([0, 310, -60, 80])

plt.scatter(df["decimalLongitude"], df["decimalLatitude"], s=8, alpha=0.1, color="red", 
            transform=ccrs.Geodetic(), rasterized=True)


# The histogram below shows the distribution of the 'day' variable.  This shows how the use of the platform has been increasing.

# In[15]:


plt.hist(2015 + df["day"] * 1000 / 365)
plt.xlabel("Year")
plt.ylabel("Frequency")


# Get the mean latitude per species.  Individuals within a species vary in terms of their locations.  We will use the mean latitude as a measure of the central latitude value for each species.

# In[16]:


meanLat = df.groupby("species")["decimalLatitude"].aggregate(np.mean)
meanLat = meanLat.rename("meanLatitude")


# In[17]:


meanLat


# The intraclass correlation (ICC) is a measure of how much the latitudes of species centroids vary in relation to how much the latitudes of individual plants vary. 

# In[18]:


df = pd.merge(df, meanLat, on="species")
icc = df["meanLatitude"].var() / df["decimalLatitude"].var()
print(icc)


# Longitude is a circular variable, we begin by converting it to a trigonometric basis.

# In[19]:


df["lonrad"] = np.pi * df["decimalLongitude"] / 180
df["lonrad_sin"] = np.sin(df["lonrad"])
df["lonrad_cos"] = np.cos(df["lonrad"])


# Create within-species residuals for the locations.

# In[21]:


for x in ["lonrad_sin", "lonrad_cos"]:
    df[x+"_cen"] = df.groupby("species")[x].transform("mean")
u = np.sqrt(df["lonrad_sin_cen"]**2 + df["lonrad_cos_cen"]**2)
for x in ["lonrad_sin", "lonrad_cos"]:
    df[x+"_cen"] /= u
    df[x+"_resid"] = df[x] - df[x+"_cen"]


# To demonstrate what this circular residualization process is doing, we make some plots below, focusing on one of the most prevalent species.

# In[22]:


ns = df["species"].value_counts()
dd = df.query("species == '%s'" % ns.index[0]).copy()

dd["nrm"] = np.sqrt(dd["lonrad_cos_resid"]**2 + dd["lonrad_sin_resid"]**2)

for vn in ["lonrad_cos_resid", "lonrad_sin_resid", "nrm"]:
    plt.figure(figsize=(7, 6))
    ax = plt.axes([0.05, 0.05, 0.84, 0.88], projection=ccrs.PlateCarree(central_longitude=180))
    ax.coastlines()
    ax.set_extent([0, 310, -60, 80])

    dd["nrmq"] = pd.qcut(dd[vn], 4)
    for (ky,d1) in dd.groupby("nrmq"):
        plt.scatter(d1["decimalLongitude"], d1["decimalLatitude"], s=8, alpha=0.9,
                    label=ky, transform=ccrs.Geodetic(), rasterized=True)
    plt.figlegend()
    plt.show()


# Create a variable that cannot contain any unique information about the outcome.  This is used to assess the validity of the analyses conducted below.

# In[23]:


df["fake"] = df["lonrad_cos"] + np.random.normal(size=df.shape[0])


# Below we fit a linear model predicting latitude from day and other variables, using OLS, for each species.  The main interest here is the relationship between "day" and the species-level mean latitude.  If this coefficient is positive for a given species, this species is identified at more northernly locations as time progresses.  If the coefficient is negative the species is identified at more southernly locations as time progresses.  We assess these effects using two models.  The first model has only main effects and the second model allows the time trend in mean latitude to vary by longitude.

# In[24]:


rr = []
for (sp,dx) in df.groupby("species"):

    if dx.shape[0] < 10:
        continue

    md1 = sm.OLS.from_formula("decimalLatitude ~ day + lonrad_sin + lonrad_cos + fake", data=dx)
    mr1 = md1.fit()

    md2 = sm.OLS.from_formula("decimalLatitude ~ day * (lonrad_sin + lonrad_cos + fake)", data=dx)
    mr2 = md2.fit()

    # The likelihood ratio test statistic and its degrees of freedom.
    lrt = 2 * (mr2.llf - mr1.llf)
    dof = mr2.df_model - mr1.df_model

    # Convert the LRT statistic to a normal score
    lrt_z = norm.ppf(chi2.cdf(lrt, dof))
    
    # This is a measure of how identifiable the model is
    ss = np.linalg.svd(md1.exog,0)[1]
    mineig = ss.min() / ss.max()
    
    # Apply a normalizing transformation to the LRT statistics.
    # This is called the Wilson-Hilferty transformation.
    if dof == 3 and mineig > 1e-7:
        lrt_zwh = (lrt / dof)**(1/3)
        lrt_zwh -= 1 - 2/(9*dof)
        lrt_zwh /= np.sqrt(2/(9*dof))
    else:
        lrt_zwh = 0

    rr.append([sp, dx.shape[0], mr1.params["day"], mr1.bse["day"], mr1.params["fake"], mr1.bse["fake"], 
               lrt_z, lrt_zwh])
 
rr = pd.DataFrame(rr, columns=["species", "n", "day_slope", "day_slope_se", "fake_slope", "fake_slope_se", 
                               "lrt_z", "lrt_zwh"])
rr = rr.loc[rr["day_slope_se"] > 0]
rr.head()


# The plot below shows that the cube root transform was very effective at normalizing the LRT statistics.

# In[25]:


plt.grid(True)
plt.plot(rr.lrt_z, rr.lrt_zwh, "o", alpha=0.5)
plt.xlabel("LRT transformed to normal score")
plt.ylabel("LRT after cube root transform")


# Since the longitude interactions are strong, we use models with interactions for the remainder of the analysis.  Furthermore, we use within species longitude residuals to control for longitude effects within each species.  After doing this, the day slopes can be interpreted as the day slopes at the central longitude of the species range.

# In[26]:


qq = []
for (sp,dx) in df.groupby("species"):

    if dx.shape[0] < 10:
        continue

    md1 = sm.OLS.from_formula("decimalLatitude ~ day + lonrad_sin_resid + lonrad_cos_resid + fake", data=dx)
    md2 = sm.OLS.from_formula("decimalLatitude ~ day * (lonrad_sin_resid + lonrad_cos_resid + fake)", data=dx)
    
    # Exclude species for which the effects are weakly identified
    ss = np.linalg.svd(md1.exog,0)[1]
    mineig = ss.min() / ss.max()
    if mineig < 1e-7:
        continue
    
    mr1 = md1.fit()
    mr2 = md2.fit()

    # The likelihood ratio test statistic and its degrees of freedom.
    lrt = 2 * (mr2.llf - mr1.llf)
    dof = mr2.df_model - mr1.df_model

    # Convert the LRT statistic to a normal score
    lrt_z = norm.ppf(chi2.cdf(lrt, dof))
    
    qq.append([sp, dx.shape[0], mr2.params["day"], mr2.bse["day"], mr2.params["fake"], mr2.bse["fake"], lrt_z])
 
qq = pd.DataFrame(qq, columns=["species", "n", "day_slope", "day_slope_se", "fake_slope", "fake_slope_se", 
                               "lrt_z"])

qq = qq.loc[rr["day_slope_se"] > 0]
qq.head()


# Construct T-scores for parameters of interest. For species with large sample sizes these should be approximate Z-scores (they follow a standard normal distribution under the null hypothesis that the day slope is zero).  For smaller sample sizes we need to account for the uncertainty in the scale parameter estimate. 

# In[27]:


qq["day_slope_t"] = qq["day_slope"] / qq["day_slope_se"]
qq["fake_slope_t"] = qq["fake_slope"] / qq["fake_slope_se"]
qq.head()


# In[28]:


qq = pd.merge(qq, meanLat, left_on="species", right_index=True)
qq.head()


# Account for finite group sizes, by mapping the t-distributed statistics to normally distributed statistics.

# In[29]:


def t_adjust(qq, vn, dof=5):
    x = tdist.cdf(qq[vn], qq["n"] - dof)
    x = np.clip(x, 1e-12, 1-1e-12)
    return norm.ppf(x)

qq["day_slope_z"] = t_adjust(qq, "day_slope_t")
qq["fake_slope_z"] = t_adjust(qq, "fake_slope_t")


# The plot below illustrates the conversion from t-scores to z-scores.

# In[30]:


plt.clf()
plt.grid(True)
plt.plot(qq["day_slope_t"], qq["day_slope_z"], "o", alpha=0.2)
plt.axline((0,0), slope=1, color="grey")
plt.xlim(-20,20)
plt.xlabel("Z-statistic", size=15)
plt.ylabel("T-statistic", size=15)


# Below we plot the distribution of day slope Z-scores.  The orange curve is what we would expect to see if no species ranges are changing.

# In[31]:


plt.hist(qq["day_slope_z"], bins=20, density=True)
x = np.linspace(-4, 4, 100)
y = np.exp(-x**2/2) / np.sqrt(2*np.pi)
plt.plot(x, y, color="orange")
plt.xlabel("Day slope (Z-score)")
plt.ylabel("Standard normal density")


# The z-scores for the estimated slope of the "fake" (simulated random) covariate match the standard normal distribution well.

# In[32]:


plt.hist(qq["fake_slope_z"], bins=20, density=True)
x = np.linspace(-4, 4, 100)
y = np.exp(-x**2/2) / np.sqrt(2*np.pi)
plt.plot(x, y, color="orange")
plt.xlabel("Fake slope (Z-score)")
plt.ylabel("Standard normal density")


# Quantile-quantile plots are usually a more informative way to compare distributions than histograms.  Below we use QQ plots to compare the Z-scores for the observed data to the reference distribution of Z-scores.  
# 
# When analyzing the "polypodiopsida" (fern) class, we see that the observed Z-scores are substantially inflated relative to the reference Z-scores, suggesting that many of the day slope parameters are substantially different from zero.  The z-scores for the "fake" variable are almost perfectly standard normal, as expected.  The third plot below shows the normalized likelihood ratio test statistic comparing the base model with additive day and longitude effects to the model with day and longitude interactions.

# In[33]:


n = qq.shape[0]
xx = np.linspace(1/n, 1-1/n, n)
yy = norm.ppf(xx)
for vn in ["day_slope", "fake_slope", "lrt"]:
    zs = np.sort(qq["%s_z" % vn])
    plt.clf()
    plt.grid(True)
    plt.plot(zs, yy, "-")
    ii = np.ceil(np.linspace(0.1, 0.9, 9) * len(yy)).astype(int)
    plt.plot(zs[ii], yy[ii], "o", color="red")
    plt.axline((0, 0), slope=1, color="grey")
    plt.xlabel("Observed %s quantiles" % vn, size=15)
    plt.ylabel("Standard normal quantiles", size=15)
    plt.show()


# To control family-wise error rates at 0.05 using the Bonferroni approach, the Z-scores must exceed the value calculated below in magnitude.

# In[34]:


n = qq["day_slope_z"].dropna().size
bonf_z = norm.ppf(1 - 0.025 / n)
np.sum(np.abs(qq["day_slope_z"]) > bonf_z)
print(n)
print(bonf_z)


# Below we plot the order statistics of the absolute values of the day slope z-scores (blue curve) in relation to the threshold that controls the family-wise error rate at 0.05 (purple line).

# In[35]:


z = np.abs(qq["day_slope_z"].dropna())
z = np.sort(z)[::-1]
plt.clf()
plt.grid(True)
plt.plot(z)
plt.axhline(bonf_z, color="purple")
plt.xlabel("Number of species")
plt.ylabel("Absolute Z-score")


# We can count the number of species that would be deemed statistically significant using the Bonferroni approach to control the FWER at 0.05.

# In[36]:


(z > bonf_z).sum()


# Below we calculate and plot the local False Discovery Rate (local FDR).

# In[37]:


qq["locfdr"] = local_fdr(qq["day_slope_z"])
lfdr = np.sort(qq["locfdr"])
lfdr = lfdr[lfdr < 1]

plt.clf()
plt.grid(True)
plt.plot(lfdr)
plt.xlabel("Number of species")
plt.ylabel("local FDR")


# We can count the number of species that would be deemed significant if we aim to control the false discovery rate to 0.1.

# In[38]:


(qq["locfdr"] <= 0.1).sum()


# Plot the day slope Z-score against the mean latitude, to assess whether there are systematic trends in the Z-scores relative to distance from the equator.  The orange curves are empirical estimates of the 10th and 90th percentiles of the Z-scores at each fixed latitude.  The purple lines are the corresponding reference values under the null hypothesis.  
# 
# When analyzing the data for "polypodiopsida" (ferns), this plot reveals several points of interest.  There is an excess of large Z-scores at every latitude, suggesting that plants are changing their distributions (in latitude terms), and this is happening at all latidues.  Second, there is a symmetry between large positive and large negative Z-scores, suggesting that plants are as likely to move toward the poles as they are to move toward the equator.  Third, there may be slightly stronger evidence for changes in the northern part of the northern hemisphere compared to other regions.

# In[39]:


alpha = 0.05

qq = qq.sort_values(by="meanLatitude")    
plt.clf()
plt.grid(True)
plt.plot(qq["meanLatitude"], qq["day_slope_z"], "o", alpha=0.5)
y1 = qq["day_slope_z"].rolling(200).quantile(1 - alpha)
y2 = qq["day_slope_z"].rolling(200).quantile(alpha)
plt.plot(qq["meanLatitude"], y1, color="orange")
plt.plot(qq["meanLatitude"], y2, color="orange")
z = norm.ppf(1 - alpha)
plt.axhline(z, color="purple")
plt.axhline(-z, color="purple")
plt.xlabel("Mean latitude", size=15)
plt.ylabel("Day slope (Z)", size=15)
plt.show()


# Next we plot the local FDR against the day slope Z-score.  This plot shows that small FDRs (<0.1) are obtained for Z-scores exceeding 3 in magnitude.

# In[40]:


plt.clf()
plt.grid(True)
plt.plot(qq["day_slope_z"], qq["locfdr"], "o", alpha=0.5)
plt.xlabel("Day slope (Z)", size=15)
plt.ylabel("Local FDR", size=15)


# Next we plot the day slope Z-score against the sample size.  If we are mainly limited by power then the larger Z-scores will be concentrated where the sample size is larger.  This plot makes it clear that there are some Z-scores falling far outside the likely range for a standard normal variable, and these values can be either positive or negative.  Most of the largest Z-scores (in magnitude) occur with the larger sample sizes.

# In[41]:


qq = qq.sort_values(by="n")
qq["logn"] = np.log10(qq["n"])
plt.clf()
plt.grid(True)
plt.plot(qq["logn"], qq["day_slope_z"], "o", alpha=0.5)
y1 = qq["day_slope_z"].rolling(200).quantile(0.05)
y2 = qq["day_slope_z"].rolling(200).quantile(0.95)
plt.plot(qq["logn"], y1, color="orange")
plt.plot(qq["logn"], y2, color="orange")
z = norm.ppf(0.95)
plt.axhline(z, color="purple")
plt.axhline(-z, color="purple")
plt.xlabel("Log10 n", size=15)
plt.ylabel("Day slope (Z)", size=15)


# We can also smooth the absolute Z-scores against log sample size.  Under the null hypothesis the Z-scores follow a standard normal distribution, and the expectation of the absolute value of a standard normal variate is $\sqrt{2/\pi}$, which is plotted below as the purple line.  It appears that there is some overdispersion of the Z-scores for the smaller sample sizes, but the extent of overdispersion (evidence for a relationship between mean latitude and time) is primarily present for the species with more than around $\exp(6) \approx 400$ observations.

# In[42]:


plt.clf()
plt.grid(True)
x = np.log(qq["n"])
y = np.abs(qq["day_slope_z"])
plt.plot(x, y, "o", alpha=0.5)
xy = lowess(y, x)
plt.plot(xy[:, 0], xy[:, 1], "-", color="orange")
plt.axhline(np.sqrt(2/np.pi), color="purple")
plt.xlabel("Log n", size=15)
plt.ylabel("Absolute day slope (|Z|)", size=15)

