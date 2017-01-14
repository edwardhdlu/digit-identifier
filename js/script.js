function regularize(r, g, b) {
	return ((r + g + b) / 3.0) / 255 - 0.5
}

function softmax(arr) {
	var sum = 0;
	var result = [];

	for (var i = 0; i < arr.length; i++) {
		sum += Math.exp(arr[i]);
	}
	for (var i = 0; i < arr.length; i++) {
		result.push(Math.exp(arr[i]) / sum);
	}

	return result;
}

$(document).ready(function() {
	var canvas = $("#canvas")[0];
	var context = canvas.getContext("2d");

	var canvas_bw = $("#canvas_bw")[0];
	var context_bw = canvas_bw.getContext("2d");

	$("#upload").change(function(e) {
		var file = e.target.files[0];
		var reader = new FileReader();

		reader.onload = function(event) {
			var img = new Image();
			img.onload = function () {
				$(".preview").show();
				$("#preview").attr("src", event.target.result);

				$("#box-2").animate({opacity: 1}, 250);
				$("#box-2 .status").hide();
				$("#box-3 .status").text("Identifying ...")

				$("html, body").animate({ scrollTop: $(document).height() }, 1000);

				var w = img.width;
				var h = img.height;
				var bound = Math.min(w, h);

				var w_off = (w - bound) / 2;
				var h_off = (h - bound) / 2;

				context.drawImage(img, w_off, h_off, bound, bound, 0, 0, 32, 32);

				var arr = [];
				for (var i = 0; i < 32; i++) {
					for (var j = 0; j < 32; j++) {
						var pixel = context.getImageData(j, i, 1, 1).data;
						var avg = parseInt((pixel[0] + pixel[1] + pixel[2]) / 3);
						var reg = regularize(pixel[0], pixel[1], pixel[2])

						context_bw.fillStyle = "rgb(" + avg + "," + avg + "," + avg + ")";
						context_bw.fillRect(j, i, 1, 1);

						arr.push(reg);
					}
				}

				// send flask request and get logits
				var url = "https://gentle-mountain-17522.herokuapp.com/predict/";
				$.ajax({
					url: url,
					type: "POST",
					data: arr.join(","),
					contentType: "text/plain"
				}).done(function(data) {
					$("#box-3").animate({opacity: 1}, 250);
					$("#box-3 .status").hide();

					var trim = data.replace("[[", "").replace("]]", "").trim();
					var result = trim.split(/\s+/g);
					var arr = result.map(function(x) { return parseFloat(x); });

					var percentages = softmax(arr);
					var sorted = softmax(arr).sort(function(a, b) { return b - a; });

					var d1 = percentages.indexOf(sorted[0]);
					var d2 = percentages.indexOf(sorted[1]);
					var d3 = percentages.indexOf(sorted[2]);

					$(".bar").show();

					var w1 = (sorted[0] * 100).toFixed(2);
					var w2 = (sorted[1] * 100).toFixed(2);
					var w3 = (sorted[2] * 100).toFixed(2);

					$("#bar-1").html("<b>" + d1 + " : </b>" + w1 + "%");
					$("#bar-2").html("<b>" + d2 + " : </b>" + w2 + "%");
					$("#bar-3").html("<b>" + d3 + " : </b>" + w3 + "%");

					$("#bar-1").width(w1 - 24 + "%");
					$("#bar-2").width(w2 - 24 + "%");
					$("#bar-3").width(w3 - 24 + "%");

					$(".prediction").show();
					$(".prediction").text(d1);
				});
			}
	        img.src = event.target.result;
	    }
		reader.readAsDataURL(file);
	});
});