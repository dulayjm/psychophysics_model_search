def get_class_to_idx():
    return {0: '00003', 1: '00020', 2: '00036', 3: '00052', 4: '00067', 5: '00086', 6: '00108', 7: '00131', 8: '00152', 9: '00168', 10: '00185', 11: '00209', 12: '00225', 13: '00241', 14: '00259', 15: '00277', 16: '00294', 17: '00314', 18: '00332', 19: '00349', 20: '00365', 21: '00383', 22: '00409', 23: '00004', 24: '00021', 25: '00037', 26: '00053', 27: '00068', 28: '00087', 29: '00109', 30: '00132', 31: '00153', 32: '00169', 33: '00186', 34: '00210', 35: '00226', 36: '00242', 37: '00260', 38: '00278', 39: '00295', 40: '00315', 41: '00333', 42: '00350', 43: '00366', 44: '00386', 45: '00410', 46: '00005', 47: '00022', 48: '00039', 49: '00054', 50: '00069', 51: '00088', 52: '00110', 53: '00133', 54: '00154', 55: '00170', 56: '00187', 57: '00211', 58: '00227', 59: '00243', 60: '00262', 61: '00279', 62: '00296', 63: '00316', 64: '00334', 65: '00351', 66: '00367', 67: '00387', 68: '00411', 69: '00006', 70: '00023', 71: '00040', 72: '00055', 73: '00070', 74: '00089', 75: '00111', 76: '00134', 77: '00155', 78: '00171', 79: '00189', 80: '00212', 81: '00228', 82: '00244', 83: '00263', 84: '00280', 85: '00297', 86: '00317', 87: '00335', 88: '00352', 89: '00368', 90: '00389', 91: '00412', 92: '00007', 93: '00024', 94: '00041', 95: '00056', 96: '00071', 97: '00090', 98: '00112', 99: '00135', 100: '00156', 101: '00172', 102: '00192', 103: '00213', 104: '00229', 105: '00245', 106: '00264', 107: '00281', 108: '00298', 109: '00318', 110: '00336', 111: '00353', 112: '00369', 113: '00390', 114: '00413', 115: '00008', 116: '00025', 117: '00042', 118: '00057', 119: '00073', 120: '00091', 121: '00113', 122: '00138', 123: '00157', 124: '00173', 125: '00195', 126: '00214', 127: '00230', 128: '00246', 129: '00265', 130: '00282', 131: '00299', 132: '00320', 133: '00337', 134: '00354', 135: '00370', 136: '00393', 137: '00009', 138: '00026', 139: '00043', 140: '00058', 141: '00074', 142: '00092', 143: '00114', 144: '00141', 145: '00158', 146: '00174', 147: '00197', 148: '00215', 149: '00231', 150: '00247', 151: '00266', 152: '00283', 153: '00302', 154: '00321', 155: '00338', 156: '00355', 157: '00372', 158: '00394', 159: '00010', 160: '00027', 161: '00044', 162: '00059', 163: '00076', 164: '00093', 165: '00115', 166: '00142', 167: '00159', 168: '00175', 169: '00199', 170: '00216', 171: '00233', 172: '00248', 173: '00269', 174: '00284', 175: '00303', 176: '00322', 177: '00339', 178: '00356', 179: '00374', 180: '00395', 181: '00011', 182: '00028', 183: '00045', 184: '00060', 185: '00078', 186: '00095', 187: '00116', 188: '00143', 189: '00161', 190: '00177', 191: '00200', 192: '00217', 193: '00234', 194: '00249', 195: '00270', 196: '00285', 197: '00304', 198: '00323', 199: '00340', 200: '00357', 201: '00375', 202: '00399', 203: '00012', 204: '00029', 205: '00046', 206: '00061', 207: '00079', 208: '00096', 209: '00120', 210: '00144', 211: '00162', 212: '00179', 213: '00202', 214: '00218', 215: '00235', 216: '00250', 217: '00271', 218: '00287', 219: '00305', 220: '00324', 221: '00341', 222: '00358', 223: '00376', 224: '00400', 225: '00013', 226: '00030', 227: '00047', 228: '00062', 229: '00080', 230: '00101', 231: '00121', 232: '00145', 233: '00163', 234: '00180', 235: '00204', 236: '00219', 237: '00236', 238: '00251', 239: '00272', 240: '00289', 241: '00306', 242: '00325', 243: '00343', 244: '00360', 245: '00377', 246: '00402', 247: '00016', 248: '00031', 249: '00048', 250: '00063', 251: '00081', 252: '00102', 253: '00122', 254: '00146', 255: '00164', 256: '00181', 257: '00205', 258: '00220', 259: '00237', 260: '00252', 261: '00273', 262: '00290', 263: '00307', 264: '00327', 265: '00344', 266: '00361', 267: '00378', 268: '00403', 269: '00017', 270: '00032', 271: '00049', 272: '00064', 273: '00083', 274: '00104', 275: '00123', 276: '00147', 277: '00165', 278: '00182', 279: '00206', 280: '00221', 281: '00238', 282: '00253', 283: '00274', 284: '00291', 285: '00308', 286: '00328', 287: '00346', 288: '00362', 289: '00379', 290: '00404', 291: '00018', 292: '00033', 293: '00050', 294: '00065', 295: '00084', 296: '00106', 297: '00127', 298: '00148', 299: '00166', 300: '00183', 301: '00207', 302: '00222', 303: '00239', 304: '00254', 305: '00275', 306: '00292', 307: '00311', 308: '00329', 309: '00347', 310: '00363', 311: '00380', 312: '00405', 313: '00019', 314: '00034', 315: '00051', 316: '00066', 317: '00085', 318: '00107', 319: '00129', 320: '00149', 321: '00167', 322: '00184', 323: '00208', 324: '00223', 325: '00240', 326: '00256', 327: '00276', 328: '00293', 329: '00312', 330: '00331', 331: '00348', 332: '00364', 333: '00381', 334: '00406'}