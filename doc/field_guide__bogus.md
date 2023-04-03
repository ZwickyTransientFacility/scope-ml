## Bogus variability (bogus)

Not all light curve variability pertains to a source's intrinsic astrophysical nature. Some is caused by nearby extended objects, bright stars, blends and image artifacts, and being aware of how such bogus light curves appear can help avoid confusion.

### ZTF light curves
![ZTF bogus](data/bogus_1.png)
![ZTF bogus](data/bogus_2.png)

#### Description
The first light curve above demonstrates a saturation ghost artifact, and the second light curve suffers from another kind of artifact. These artifacts were identified and affected data masked after ZTF began, but data in earlier releases (such as the above from DR5) were not retroactively masked. This produces the apparent cutoff in variability after a certain point in time.

#### Light curve characteristics
The light curves appear to be "flaring" with departures from the median by multiple magnitudes which suddenly stop in later data. This cutoff corresponds to the time when a new method of processing the data was used to mask affected points.

### References and further reading:
- [ZTF Explanatory Supplement (esp. Appendix B)](https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf)
