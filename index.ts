takePhoto: function() {
  const that = this;

  wx.chooseImage({
    count: 9,
    sizeType: ['original', 'compressed'],
    sourceType: ['camera', 'album'],
    success: function(res) {
      const tempFilePaths = res.tempFilePaths;
      const currentTime = new Date().toLocaleString();

      // 获取位置信息
      wx.getLocation({
        type: 'wgs84',
        success: (locRes) => {
          that.setData({
            imageSrc: tempFilePaths,
            timestamp: currentTime,
            location: `${locRes.latitude.toFixed(4)}, ${locRes.longitude.toFixed(4)}`
          });
        },
        fail: () => {
          that.setData({
            imageSrc: tempFilePaths,
            timestamp: currentTime,
            location: '位置获取失败'
          });
        }
      });
    },
    fail: (err) => {
      wx.showToast({
        title: '选择失败，请重试',
        icon: 'none'
      });
      console.error('媒体选择错误：', err);
    }
  });
},