import { Component, OnInit } from '@angular/core';
import { CaptureImageOptions, MediaCapture } from '@awesome-cordova-plugins/media-capture/ngx';
@Component({
  selector: 'app-issue',
  templateUrl: './issue.page.html',
  styleUrls: ['./issue.page.scss'],
})
export class IssuePage implements OnInit {

  video:any;
  constructor(private mediaCapture:MediaCapture) { }

  ngOnInit() {
  }

  async startRecording(){
    try {
      let options:CaptureImageOptions={limit:1}
      const data=await this.mediaCapture.captureVideo(options)!;
      console.log(data);
      this.video=data;
      this.video=this.video[0];
      console.log(this.video);
    } catch (e:any) {
      console.log(e);
      
    }
  }

}
